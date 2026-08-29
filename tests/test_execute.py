"""Execution tests.

The claims under test, in order of how badly they would hurt if false:

* The same logical action never reaches the provider twice.
* A crash between issuing and recording leaves a recoverable trace.
* Unmapped provider errors are not retried.
* A notification missing a legally required field is not sent, however well
  it reads.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from recovery.domain.actions import Action
from recovery.domain.entities import Mandate
from recovery.domain.enums import ActionType, Channel, MandateCategory
from recovery.execute.client import (
    DuplicateOperationError,
    ReplayClient,
    classify_provider_error,
)
from recovery.execute.executor import MAX_PROVIDER_RETRIES, Executor
from recovery.execute.idempotency import IdempotencyLedger, idempotency_key
from recovery.execute.notifications import (
    NotificationValidationError,
    compose,
    render_template,
    validate,
)

NOW = datetime(2026, 7, 6, 6, 30, tzinfo=UTC)

REQUIRED_FIELDS = [
    "merchant_name",
    "amount",
    "debit_datetime",
    "mandate_reference",
    "debit_reason",
    "opt_out_instruction",
]

LINK = Action(action_type=ActionType.SEND_PAYMENT_LINK, channel=Channel.SMS)
RETRY = Action(action_type=ActionType.RETRY_NOW)


def _fixture(*records: dict[str, Any]) -> ReplayClient:
    return ReplayClient(list(records))


def _ok(op: str, **response: Any) -> dict[str, Any]:
    return {"op": op, "request": {}, "response": response, "error": None}


def _err(op: str, code: str, description: str = "boom") -> dict[str, Any]:
    return {
        "op": op,
        "request": {},
        "response": None,
        "error": {"code": code, "description": description},
    }


@pytest.fixture
def ledger(tmp_path: Path) -> IdempotencyLedger:
    return IdempotencyLedger(tmp_path / "ledger.sqlite3")


def _mandate() -> Mandate:
    return Mandate(
        mandate_id="mdt_abc123",
        customer_id="cst_1",
        merchant_id="mrc_1",
        category=MandateCategory.SUBSCRIPTION,
        max_amount_paise=99800,
        debit_amount_paise=49900,
        registered_at=NOW - timedelta(days=90),
        valid_until=NOW + timedelta(days=365),
        next_debit_at=NOW + timedelta(days=1),
    )


class TestIdempotencyKeys:
    def test_key_is_deterministic(self) -> None:
        """A fresh key per attempt would make every retry look like a new
        charge — which is the bug, not the fix."""
        args = {
            "case_id": "case_1",
            "action": ActionType.RETRY_NOW,
            "attempt": 1,
            "amount_paise": 49900,
        }
        assert idempotency_key(**args) == idempotency_key(**args)  # type: ignore[arg-type]

    def test_key_excludes_wall_clock(self) -> None:
        """Two attempts on the same operation hours apart are the same
        operation. Including time would defeat the whole mechanism."""
        first = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=1, amount_paise=100
        )
        second = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=1, amount_paise=100
        )
        assert first == second

    def test_amount_change_produces_a_new_key(self) -> None:
        """A re-quoted amount is a different operation and must not be
        swallowed by a stale key."""
        base = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=1, amount_paise=100
        )
        changed = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=1, amount_paise=200
        )
        assert base != changed

    def test_attempt_number_produces_a_new_key(self) -> None:
        base = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=1, amount_paise=100
        )
        second = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=2, amount_paise=100
        )
        assert base != second


class TestNoDoubleCharge:
    def test_repeated_execution_calls_provider_once(self, ledger: IdempotencyLedger) -> None:
        client = _fixture(
            _ok(
                "create_payment_link",
                id="plink_1",
                status="created",
                short_url="https://rzp.io/l/1",
            ),
            _ok(
                "create_payment_link",
                id="plink_2",
                status="created",
                short_url="https://rzp.io/l/2",
            ),
        )
        executor = Executor(client, ledger, sleep=lambda _s: None)
        args = {"case_id": "case_1", "action": LINK, "amount_paise": 49900, "attempt": 1}

        first = executor.execute(**args)  # type: ignore[arg-type]
        second = executor.execute(**args)  # type: ignore[arg-type]

        assert first.provider_reference == "plink_1"
        # The second call must return the *original* reference, not plink_2.
        assert second.provider_reference == "plink_1"
        assert second.skipped_reason is not None
        assert len(ledger.for_case("case_1")) == 1

    def test_dedup_is_recorded_not_silent(self, ledger: IdempotencyLedger) -> None:
        """'We did not double-charge' needs evidence like any other claim."""
        client = _fixture(_ok("create_order", id="order_1", status="created"))
        executor = Executor(client, ledger, sleep=lambda _s: None)
        args = {"case_id": "c", "action": RETRY, "amount_paise": 100, "attempt": 1}
        executor.execute(**args)  # type: ignore[arg-type]
        second = executor.execute(**args)  # type: ignore[arg-type]
        assert second.record.api_endpoint == "deduplicated"
        assert "already succeeded" in (second.skipped_reason or "")

    def test_reservation_precedes_the_call(self, ledger: IdempotencyLedger) -> None:
        """A crash mid-flight must leave a trace. Writing the row after the
        call would leave none, and the operation would repeat on restart."""

        class Exploding:
            def create_order(self, **_: Any) -> dict[str, Any]:
                raise RuntimeError("connection reset")

            def create_payment_link(self, **_: Any) -> dict[str, Any]:
                raise RuntimeError("connection reset")

            def fetch_order(self, order_id: str) -> dict[str, Any]:
                raise RuntimeError("connection reset")

        executor = Executor(Exploding(), ledger, sleep=lambda _s: None)
        with pytest.raises(RuntimeError):
            executor.execute(case_id="case_x", action=RETRY, amount_paise=100, attempt=1)
        in_flight = ledger.in_flight()
        assert len(in_flight) == 1
        assert in_flight[0].case_id == "case_x"

    def test_attempts_counted_from_ledger(self, ledger: IdempotencyLedger) -> None:
        """In-memory case state can be stale after a crash; the ledger is what
        actually reached the provider."""
        client = _fixture(
            _ok("create_order", id="o1", status="created"),
            _ok("create_order", id="o2", status="created"),
        )
        executor = Executor(client, ledger, sleep=lambda _s: None)
        executor.execute(case_id="c", action=RETRY, amount_paise=100, attempt=1)
        executor.execute(case_id="c", action=RETRY, amount_paise=100, attempt=2)
        assert ledger.attempts_used("c") == 2


class TestErrorHandling:
    def test_retriable_error_is_retried(self, ledger: IdempotencyLedger) -> None:
        client = _fixture(
            _err("create_order", "GATEWAY_ERROR"),
            _err("create_order", "GATEWAY_ERROR"),
            _err("create_order", "GATEWAY_ERROR"),
        )
        executor = Executor(client, ledger, sleep=lambda _s: None)
        result = executor.execute(case_id="c", action=RETRY, amount_paise=100, attempt=1)
        assert not result.succeeded
        assert result.record.retry_count == MAX_PROVIDER_RETRIES

    def test_non_retriable_error_is_not_retried(self, ledger: IdempotencyLedger) -> None:
        """Retrying a malformed request cannot succeed and only delays
        escalation."""
        client = _fixture(_err("create_order", "BAD_REQUEST_ERROR", "amount invalid"))
        executor = Executor(client, ledger, sleep=lambda _s: None)
        result = executor.execute(case_id="c", action=RETRY, amount_paise=100, attempt=1)
        assert not result.succeeded
        assert result.record.retry_count == 0

    def test_unknown_error_defaults_to_not_retriable(self) -> None:
        """Assuming an unfamiliar failure is safe to repeat is the reasoning
        that produces duplicate charges."""
        error = classify_provider_error("SOME_NEW_CODE_2027", "unfamiliar")
        assert not error.retriable

    def test_failure_is_recorded_with_the_provider_code(self, ledger: IdempotencyLedger) -> None:
        client = _fixture(_err("create_order", "BAD_REQUEST_ERROR", "amount invalid"))
        executor = Executor(client, ledger, sleep=lambda _s: None)
        result = executor.execute(case_id="c", action=RETRY, amount_paise=100, attempt=1)
        assert "BAD_REQUEST_ERROR" in (result.record.error_reason or "")

    def test_rate_limit_is_retriable(self) -> None:
        assert classify_provider_error("RATE_LIMIT_ERROR", "slow down").retriable


class TestLiveClientSafety:
    def test_refuses_a_non_test_key(self) -> None:
        from recovery.execute.client import LiveClient

        with pytest.raises(ValueError, match="test mode"):
            LiveClient("rzp_live_abc123", "secret")


class TestNotifications:
    def test_template_satisfies_every_required_field(self) -> None:
        content = render_template(
            merchant_name="Acme Streaming",
            mandate=_mandate(),
            debit_at=NOW + timedelta(days=1),
            debit_reason="monthly subscription",
        )
        assert validate(content, REQUIRED_FIELDS) == []

    def test_missing_opt_out_is_detected(self) -> None:
        """The failure mode that reads perfectly well and is still illegal."""
        content = render_template(
            merchant_name="Acme Streaming",
            mandate=_mandate(),
            debit_at=NOW + timedelta(days=1),
            debit_reason="monthly subscription",
        )
        stripped = content.body.replace(content.opt_out_instruction, "")
        broken = type(content)(**{**content.__dict__, "body": stripped})
        assert "opt_out_instruction" in validate(broken, REQUIRED_FIELDS)

    def test_missing_amount_is_detected(self) -> None:
        content = render_template(
            merchant_name="Acme Streaming",
            mandate=_mandate(),
            debit_at=NOW + timedelta(days=1),
            debit_reason="monthly subscription",
        )
        broken = type(content)(**{**content.__dict__, "body": "We will debit you soon."})
        missing = validate(broken, REQUIRED_FIELDS)
        assert "amount" in missing
        assert "mandate_reference" in missing

    def test_compose_raises_when_template_diverges_from_policy(self) -> None:
        """If a field is added to policy.yaml that the template does not
        render, that must break loudly rather than ship silently."""
        with pytest.raises(NotificationValidationError):
            compose(
                merchant_name="Acme",
                mandate=_mandate(),
                debit_at=NOW + timedelta(days=1),
                debit_reason="subscription",
                required_fields=[*REQUIRED_FIELDS, "customer_grievance_contact"],
            )

    def test_compose_returns_template_without_llm(self) -> None:
        content = compose(
            merchant_name="Acme",
            mandate=_mandate(),
            debit_at=NOW + timedelta(days=1),
            debit_reason="subscription",
            required_fields=REQUIRED_FIELDS,
        )
        assert content.drafted_by == "template"

    def test_notification_carries_mandate_reference(self) -> None:
        content = render_template(
            merchant_name="Acme",
            mandate=_mandate(),
            debit_at=NOW + timedelta(days=1),
            debit_reason="subscription",
        )
        assert "mdt_abc123" in content.body


class TestExecutionRecords:
    def test_inaction_is_still_recorded(self, ledger: IdempotencyLedger) -> None:
        """There is no silent path. Choosing to do nothing is a decision with
        a record, not an absence of one."""
        executor = Executor(_fixture(), ledger, sleep=lambda _s: None)
        result = executor.execute(
            case_id="c",
            action=Action(action_type=ActionType.NO_ACTION),
            amount_paise=100,
            attempt=0,
        )
        assert result.succeeded
        assert result.skipped_reason == "policy chose inaction"

    def test_payment_link_url_is_surfaced(self, ledger: IdempotencyLedger) -> None:
        client = _fixture(
            _ok(
                "create_payment_link",
                id="plink_9",
                status="created",
                short_url="https://rzp.io/l/demo",
            )
        )
        executor = Executor(client, ledger, sleep=lambda _s: None)
        result = executor.execute(case_id="c", action=LINK, amount_paise=49900, attempt=1)
        assert result.payment_link_url == "https://rzp.io/l/demo"

    def test_idempotency_key_is_the_order_receipt(self, ledger: IdempotencyLedger) -> None:
        """Passing the key as the receipt makes Razorpay itself reject a
        duplicate, rather than relying only on our ledger."""
        seen: dict[str, Any] = {}

        class Capturing:
            def create_order(self, **kw: Any) -> dict[str, Any]:
                seen.update(kw)
                return {"id": "order_1", "status": "created"}

            def create_payment_link(self, **kw: Any) -> dict[str, Any]:
                return {"id": "plink_1", "status": "created"}

            def fetch_order(self, order_id: str) -> dict[str, Any]:
                return {}

        executor = Executor(Capturing(), ledger, sleep=lambda _s: None)
        executor.execute(case_id="c", action=RETRY, amount_paise=100, attempt=1)
        expected = idempotency_key(
            case_id="c", action=ActionType.RETRY_NOW, attempt=1, amount_paise=100
        )
        assert seen["receipt"] == expected


DUPLICATE_DESCRIPTION = (
    "payment link with given reference_id: demo_001 already exists. "
    "Please create a payment link with a different reference_id"
)


class TestDuplicateOperations:
    """Regression cover for INC-022, found by a live call rather than a test.

    Razorpay returns a generic BAD_REQUEST_ERROR when refusing an operation it
    already holds, so the code alone cannot distinguish "you already did this"
    from "your request was malformed" — opposite meanings with opposite
    correct responses.
    """

    def test_duplicate_is_its_own_error_type(self) -> None:
        error = classify_provider_error("BAD_REQUEST_ERROR", DUPLICATE_DESCRIPTION)
        assert isinstance(error, DuplicateOperationError)

    def test_existing_reference_is_recovered(self) -> None:
        """The reference is the whole value of the response: it tells us which
        earlier operation this collides with."""
        error = classify_provider_error("BAD_REQUEST_ERROR", DUPLICATE_DESCRIPTION)
        assert isinstance(error, DuplicateOperationError)
        assert error.existing_reference == "demo_001"

    def test_genuine_bad_request_is_not_a_duplicate(self) -> None:
        error = classify_provider_error("BAD_REQUEST_ERROR", "amount must be at least INR 1.00")
        assert not isinstance(error, DuplicateOperationError)
        assert not error.retriable

    def test_duplicate_is_never_retried(self) -> None:
        error = classify_provider_error("BAD_REQUEST_ERROR", DUPLICATE_DESCRIPTION)
        assert not error.retriable

    def test_executor_treats_duplicate_as_success(self, ledger: IdempotencyLedger) -> None:
        """Escalating here would chase a case the provider already handled."""
        client = _fixture(_err("create_payment_link", "BAD_REQUEST_ERROR", DUPLICATE_DESCRIPTION))
        executor = Executor(client, ledger, sleep=lambda _s: None)
        result = executor.execute(case_id="c", action=LINK, amount_paise=49900, attempt=1)
        assert result.succeeded
        assert result.provider_reference == "demo_001"

    def test_duplicate_is_recorded_as_succeeded_in_the_ledger(
        self, ledger: IdempotencyLedger
    ) -> None:
        client = _fixture(_err("create_payment_link", "BAD_REQUEST_ERROR", DUPLICATE_DESCRIPTION))
        executor = Executor(client, ledger, sleep=lambda _s: None)
        executor.execute(case_id="c", action=LINK, amount_paise=49900, attempt=1)
        entry = ledger.for_case("c")[0]
        assert entry.status == "succeeded"
        assert entry.is_terminal

    def test_duplicate_reason_is_surfaced_not_silent(self, ledger: IdempotencyLedger) -> None:
        """Success by way of an earlier attempt is materially different from
        success by way of this one, and the trail must say which."""
        client = _fixture(_err("create_payment_link", "BAD_REQUEST_ERROR", DUPLICATE_DESCRIPTION))
        executor = Executor(client, ledger, sleep=lambda _s: None)
        result = executor.execute(case_id="c", action=LINK, amount_paise=49900, attempt=1)
        assert "already exists" in (result.skipped_reason or "")
