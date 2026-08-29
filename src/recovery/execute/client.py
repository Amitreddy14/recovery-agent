"""Razorpay test-mode client.

Three implementations behind one protocol:

* **LiveClient** — the real SDK against test-mode keys. Creates real orders
  and real payment links with openable URLs.
* **RecordingClient** — wraps Live and writes every response to a fixture
  file.
* **ReplayClient** — serves those fixtures with no network.

The reason for the split is that CI has no network and no credentials, so a
test suite that called the live API would either be skipped in CI (making it
decorative) or would embed secrets (making it a liability). Record once,
replay forever: the tests then exercise the *real* response shapes, including
the error ones, without a network dependency.

This also makes the error paths testable at all. Provoking a genuine gateway
timeout on demand is impractical; replaying a recorded one is trivial, and
error handling that has never been executed is not error handling.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from recovery.domain.enums import ErrorSource, FailureReason, PaymentStep


class ProviderError(RuntimeError):
    """A structured failure from the payment provider."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        source: ErrorSource,
        step: PaymentStep,
        reason: FailureReason,
        retriable: bool,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.source = source
        self.step = step
        self.reason = reason
        self.retriable = retriable
        self.http_status = http_status


class DuplicateOperationError(ProviderError):
    """The provider already holds this operation.

    Not a failure. It means an earlier request with the same reference
    succeeded, so this is provider-side idempotency working exactly as
    intended (ADR-0020). Conflating it with a malformed request would make the
    executor escalate a case that has already been handled — opposite meanings
    with opposite correct responses (INC-022).

    Found by a live test-mode call, not by the test suite: the condition only
    arises against a real account holding real prior state.
    """

    def __init__(self, message: str, *, code: str, existing_reference: str | None):
        super().__init__(
            message,
            code=code,
            source=ErrorSource.BUSINESS,
            step=PaymentStep.PAYMENT_INITIATION,
            reason=FailureReason.OTHER,
            retriable=False,
        )
        self.existing_reference = existing_reference


class RazorpayClient(Protocol):
    def create_order(
        self, *, amount_paise: int, receipt: str, notes: dict[str, str]
    ) -> dict[str, Any]: ...

    def create_payment_link(
        self,
        *,
        amount_paise: int,
        reference_id: str,
        description: str,
        notes: dict[str, str],
    ) -> dict[str, Any]: ...

    def fetch_order(self, order_id: str) -> dict[str, Any]: ...


# --- error mapping ---------------------------------------------------------
# Razorpay's error codes mapped onto our domain taxonomy. Anything unmapped
# becomes OTHER and non-retriable: guessing that an unknown error is safe to
# retry is how a novel failure turns into a double charge.

ERROR_MAP: dict[str, tuple[ErrorSource, PaymentStep, FailureReason, bool]] = {
    "GATEWAY_ERROR": (
        ErrorSource.GATEWAY,
        PaymentStep.PAYMENT_AUTHORIZATION,
        FailureReason.GATEWAY_TECHNICAL_ERROR,
        True,
    ),
    "SERVER_ERROR": (
        ErrorSource.GATEWAY,
        PaymentStep.PAYMENT_INITIATION,
        FailureReason.GATEWAY_TECHNICAL_ERROR,
        True,
    ),
    "BAD_REQUEST_ERROR": (
        ErrorSource.BUSINESS,
        PaymentStep.PAYMENT_INITIATION,
        FailureReason.OTHER,
        False,
    ),
    "RATE_LIMIT_ERROR": (
        ErrorSource.GATEWAY,
        PaymentStep.PAYMENT_INITIATION,
        FailureReason.OTHER,
        True,
    ),
}


# Substrings Razorpay uses when refusing an operation it already holds. Matched
# on the description because the provider returns a generic BAD_REQUEST_ERROR
# code for this, which is indistinguishable from a genuinely malformed request
# at the code level.
DUPLICATE_MARKERS: tuple[str, ...] = (
    "already exists",
    "duplicate",
)

REFERENCE_PATTERN = re.compile(r"reference_id:\s*([A-Za-z0-9_\-]+)")


def looks_like_duplicate(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in DUPLICATE_MARKERS)


def classify_provider_error(code: str, description: str) -> ProviderError:
    """Map a provider failure onto the domain taxonomy.

    Duplicate rejections are separated out first. Anything unrecognised is
    non-retriable by default: assuming an unfamiliar failure is safe to repeat
    is the reasoning that produces double charges (ADR-0022).
    """
    if looks_like_duplicate(description):
        match = REFERENCE_PATTERN.search(description)
        return DuplicateOperationError(
            description or code,
            code=code,
            existing_reference=match.group(1) if match else None,
        )

    source, step, reason, retriable = ERROR_MAP.get(
        code,
        (ErrorSource.GATEWAY, PaymentStep.PAYMENT_INITIATION, FailureReason.OTHER, False),
    )
    return ProviderError(
        description or code,
        code=code,
        source=source,
        step=step,
        reason=reason,
        retriable=retriable,
    )


class LiveClient:
    """Real Razorpay SDK against test-mode credentials.

    Refuses to start on a live key. The check is cheap and the failure it
    prevents is not.
    """

    def __init__(self, key_id: str, key_secret: str) -> None:
        if not key_id.startswith("rzp_test_"):
            raise ValueError(
                f"refusing to run against non-test key {key_id[:12]}...; "
                "this project only ever touches test mode"
            )
        import razorpay

        self._client = razorpay.Client(auth=(key_id, key_secret))
        self._client.set_app_details({"title": "recovery-agent", "version": "0.8"})

    @classmethod
    def from_env(cls) -> LiveClient:
        key_id = os.environ.get("RAZORPAY_KEY_ID", "")
        secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
        if not key_id or not secret:
            raise RuntimeError(
                "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set; see .env.example"
            )
        return cls(key_id, secret)

    def create_order(
        self, *, amount_paise: int, receipt: str, notes: dict[str, str]
    ) -> dict[str, Any]:
        return self._call(
            lambda: self._client.order.create(
                {
                    "amount": amount_paise,
                    "currency": "INR",
                    # Razorpay enforces receipt uniqueness, so passing the
                    # idempotency key here makes the provider itself reject a
                    # duplicate rather than relying only on our ledger.
                    "receipt": receipt,
                    "notes": notes,
                }
            )
        )

    def create_payment_link(
        self,
        *,
        amount_paise: int,
        reference_id: str,
        description: str,
        notes: dict[str, str],
    ) -> dict[str, Any]:
        return self._call(
            lambda: self._client.payment_link.create(
                {
                    "amount": amount_paise,
                    "currency": "INR",
                    "description": description,
                    "reference_id": reference_id,
                    "notes": notes,
                    "reminder_enable": False,
                }
            )
        )

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        return self._call(lambda: self._client.order.fetch(order_id))

    @staticmethod
    def _call(fn: Any) -> dict[str, Any]:
        try:
            return dict(fn())
        except Exception as exc:
            code = getattr(exc, "code", None) or type(exc).__name__
            raise classify_provider_error(str(code), str(exc)) from exc


class RecordingClient:
    """Wraps a live client and writes every response to a fixture."""

    def __init__(self, inner: RazorpayClient, fixture_path: Path) -> None:
        self.inner = inner
        self.fixture_path = fixture_path
        self._records: list[dict[str, Any]] = []

    def _record(self, op: str, request: dict[str, Any]) -> dict[str, Any]:
        try:
            response = getattr(self.inner, op)(**request)
            self._records.append(
                {"op": op, "request": request, "response": response, "error": None}
            )
            return dict(response)
        except ProviderError as exc:
            self._records.append(
                {
                    "op": op,
                    "request": request,
                    "response": None,
                    "error": {"code": exc.code, "description": str(exc)},
                }
            )
            raise
        finally:
            self.fixture_path.parent.mkdir(parents=True, exist_ok=True)
            self.fixture_path.write_text(
                json.dumps(self._records, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

    def create_order(
        self, *, amount_paise: int, receipt: str, notes: dict[str, str]
    ) -> dict[str, Any]:
        return self._record(
            "create_order",
            {"amount_paise": amount_paise, "receipt": receipt, "notes": notes},
        )

    def create_payment_link(
        self,
        *,
        amount_paise: int,
        reference_id: str,
        description: str,
        notes: dict[str, str],
    ) -> dict[str, Any]:
        return self._record(
            "create_payment_link",
            {
                "amount_paise": amount_paise,
                "reference_id": reference_id,
                "description": description,
                "notes": notes,
            },
        )

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        return self._record("fetch_order", {"order_id": order_id})


class ReplayClient:
    """Serves recorded fixtures. No network, no credentials.

    Matches on operation name and cycles through recordings in order, so a
    fixture containing a success followed by a gateway error reproduces that
    exact sequence — which is how the retry path gets tested.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._cursor: dict[str, int] = {}

    @classmethod
    def from_file(cls, path: Path) -> ReplayClient:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _next(self, op: str) -> dict[str, Any]:
        matching = [r for r in self._records if r["op"] == op]
        if not matching:
            raise LookupError(f"no recorded response for {op}")
        i = self._cursor.get(op, 0)
        record = matching[min(i, len(matching) - 1)]
        self._cursor[op] = i + 1
        if record["error"]:
            raise classify_provider_error(record["error"]["code"], record["error"]["description"])
        return dict(record["response"])

    def create_order(
        self, *, amount_paise: int, receipt: str, notes: dict[str, str]
    ) -> dict[str, Any]:
        return self._next("create_order")

    def create_payment_link(
        self,
        *,
        amount_paise: int,
        reference_id: str,
        description: str,
        notes: dict[str, str],
    ) -> dict[str, Any]:
        return self._next("create_payment_link")

    def fetch_order(self, order_id: str) -> dict[str, Any]:
        return self._next("fetch_order")
