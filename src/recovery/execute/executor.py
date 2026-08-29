"""The executor.

Takes a gated decision and makes it happen against Razorpay test mode, or
explains precisely why it did not.

Every path through this module writes an `ExecutionRecord`. There is no
silent success and no silent failure — an action that was skipped because the
ledger already had it is recorded as such, because "we did not double-charge"
is a claim that needs evidence like any other.

Retry policy is deliberately narrow. Only errors classified retriable are
retried, with exponential backoff and a hard attempt cap. An unmapped error
is not retried: assuming an unfamiliar failure is safe to repeat is the
reasoning that produces duplicate charges.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from recovery.domain.actions import Action, ExecutionRecord
from recovery.domain.entities import Mandate
from recovery.domain.enums import (
    ActionType,
    ErrorSource,
    FailureReason,
    PaymentStep,
)
from recovery.execute.client import (
    DuplicateOperationError,
    ProviderError,
    RazorpayClient,
)
from recovery.execute.idempotency import IdempotencyLedger, idempotency_key
from recovery.execute.notifications import NotificationContent

# Fallback classification for errors we raise ourselves rather than receive.
_UNKNOWN_SOURCE = ErrorSource.BUSINESS
_UNKNOWN_STEP = PaymentStep.PAYMENT_INITIATION
_UNKNOWN_REASON = FailureReason.OTHER

MAX_PROVIDER_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0


@dataclass(frozen=True)
class ExecutionResult:
    record: ExecutionRecord
    provider_reference: str | None = None
    payment_link_url: str | None = None
    notification: NotificationContent | None = None
    skipped_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.record.succeeded


class Executor:
    def __init__(
        self,
        client: RazorpayClient,
        ledger: IdempotencyLedger,
        *,
        merchant_name: str = "Recovery Demo Merchant",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.merchant_name = merchant_name
        self._sleep = sleep

    def execute(
        self,
        *,
        case_id: str,
        action: Action,
        amount_paise: int,
        attempt: int,
        mandate: Mandate | None = None,
        notification: NotificationContent | None = None,
    ) -> ExecutionResult:
        now = datetime.now(UTC)
        key = idempotency_key(
            case_id=case_id,
            action=action.action_type,
            attempt=attempt,
            amount_paise=amount_paise,
        )

        if action.action_type is ActionType.NO_ACTION:
            return self._record(
                case_id,
                action,
                key,
                now,
                True,
                endpoint="none",
                skipped_reason="policy chose inaction",
            )

        # Claim the key before touching the network. An existing row means
        # this exact operation has already been issued, and reissuing it is
        # the double-charge bug.
        existing = self.ledger.reserve(
            key=key,
            case_id=case_id,
            action=action.action_type,
            attempt=attempt,
            amount_paise=amount_paise,
        )
        if existing is not None:
            return self._record(
                case_id,
                action,
                key,
                now,
                existing.status == "succeeded",
                endpoint="deduplicated",
                skipped_reason=(
                    f"idempotency key already {existing.status}"
                    f"{f' as {existing.provider_reference}' if existing.provider_reference else ''}"
                ),
                provider_reference=existing.provider_reference,
            )

        try:
            return self._dispatch(
                case_id=case_id,
                action=action,
                key=key,
                amount_paise=amount_paise,
                now=now,
                mandate=mandate,
                notification=notification,
            )
        except DuplicateOperationError as exc:
            # The provider already holds this operation, so it succeeded
            # earlier. This is provider-side idempotency working, not a
            # failure — escalating here would chase a case already handled
            # (INC-022).
            self.ledger.complete(
                key=key,
                status="succeeded",
                provider_reference=exc.existing_reference,
            )
            return self._record(
                case_id,
                action,
                key,
                now,
                True,
                endpoint=_endpoint_for(action.action_type),
                provider_reference=exc.existing_reference,
                skipped_reason=(
                    "provider reports this operation already exists"
                    f"{f' as {exc.existing_reference}' if exc.existing_reference else ''}"
                ),
            )
        except ProviderError as exc:
            self.ledger.complete(
                key=key,
                status="failed_permanent" if not exc.retriable else "failed_retriable",
            )
            return self._record(
                case_id,
                action,
                key,
                now,
                False,
                endpoint=_endpoint_for(action.action_type),
                error_reason=f"{exc.code}: {exc}",
                retry_count=MAX_PROVIDER_RETRIES if exc.retriable else 0,
            )

    # -- dispatch ----------------------------------------------------------

    def _dispatch(
        self,
        *,
        case_id: str,
        action: Action,
        key: str,
        amount_paise: int,
        now: datetime,
        mandate: Mandate | None,
        notification: NotificationContent | None,
    ) -> ExecutionResult:
        notes = {"case_id": case_id, "action": action.action_type.value}

        if action.action_type in (
            ActionType.RETRY_NOW,
            ActionType.RETRY_SCHEDULED,
            ActionType.RETRY_ALTERNATE_RAIL,
        ):
            response, retries = self._with_backoff(
                lambda: self.client.create_order(
                    amount_paise=amount_paise, receipt=key, notes=notes
                )
            )
            reference = str(response.get("id", ""))
            self.ledger.complete(
                key=key,
                status="succeeded",
                provider_reference=reference,
                response=response,
            )
            return self._record(
                case_id,
                action,
                key,
                now,
                True,
                endpoint="orders.create",
                provider_reference=reference,
                retry_count=retries,
                response_code=str(response.get("status", "")),
            )

        if action.action_type is ActionType.SEND_PAYMENT_LINK:
            response, retries = self._with_backoff(
                lambda: self.client.create_payment_link(
                    amount_paise=amount_paise,
                    reference_id=key,
                    description=f"Payment for case {case_id}",
                    notes=notes,
                )
            )
            reference = str(response.get("id", ""))
            self.ledger.complete(
                key=key,
                status="succeeded",
                provider_reference=reference,
                response=response,
            )
            return self._record(
                case_id,
                action,
                key,
                now,
                True,
                endpoint="payment_link.create",
                provider_reference=reference,
                retry_count=retries,
                response_code=str(response.get("status", "")),
                payment_link_url=str(response.get("short_url", "")) or None,
            )

        if action.action_type is ActionType.PRE_DEBIT_NUDGE:
            # No SMS gateway in scope. The compliance-relevant artifact is the
            # validated content and the timestamp, both of which are recorded;
            # sending is a delivery concern, not a decision concern.
            if notification is None:
                raise ProviderError(
                    "pre-debit notification requires validated content",
                    code="BAD_REQUEST_ERROR",
                    source=_UNKNOWN_SOURCE,
                    step=_UNKNOWN_STEP,
                    reason=_UNKNOWN_REASON,
                    retriable=False,
                )
            self.ledger.complete(
                key=key, status="succeeded", provider_reference=f"notice_{key[-8:]}"
            )
            return self._record(
                case_id,
                action,
                key,
                now,
                True,
                endpoint="notification.compose",
                provider_reference=f"notice_{key[-8:]}",
                notification=notification,
            )

        if action.action_type is ActionType.ESCALATE_HUMAN:
            self.ledger.complete(
                key=key, status="succeeded", provider_reference=f"queue_{key[-8:]}"
            )
            return self._record(
                case_id,
                action,
                key,
                now,
                True,
                endpoint="queue.enqueue",
                provider_reference=f"queue_{key[-8:]}",
            )

        raise ProviderError(
            f"no execution path for {action.action_type.value}",
            code="BAD_REQUEST_ERROR",
            source=_UNKNOWN_SOURCE,
            step=_UNKNOWN_STEP,
            reason=_UNKNOWN_REASON,
            retriable=False,
        )

    # -- retry -------------------------------------------------------------

    def _with_backoff(self, call: Callable[[], dict[str, Any]]) -> tuple[dict[str, object], int]:
        """Retry only what is classified retriable, with exponential backoff.

        A non-retriable error propagates on the first failure. Retrying a
        BAD_REQUEST_ERROR cannot succeed and only delays the escalation.
        """
        delay = BASE_BACKOFF_SECONDS
        for attempt in range(MAX_PROVIDER_RETRIES):
            try:
                return dict(call()), attempt
            except ProviderError as exc:
                if not exc.retriable or attempt == MAX_PROVIDER_RETRIES - 1:
                    raise
                self._sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF_SECONDS)
        # Only reachable if MAX_PROVIDER_RETRIES is zero, which would mean the
        # executor is configured never to call the provider at all.
        raise AssertionError(f"MAX_PROVIDER_RETRIES is {MAX_PROVIDER_RETRIES}; must be >= 1")

    # -- recording ---------------------------------------------------------

    @staticmethod
    def _record(
        case_id: str,
        action: Action,
        key: str,
        now: datetime,
        succeeded: bool,
        *,
        endpoint: str,
        provider_reference: str | None = None,
        response_code: str | None = None,
        error_reason: str | None = None,
        retry_count: int = 0,
        skipped_reason: str | None = None,
        payment_link_url: str | None = None,
        notification: NotificationContent | None = None,
    ) -> ExecutionResult:
        record = ExecutionRecord(
            case_id=case_id,
            idempotency_key=key,
            action_type=action.action_type,
            executed_at=now,
            api_endpoint=endpoint,
            request_digest=key[-16:],
            succeeded=succeeded,
            response_code=response_code,
            error_reason=error_reason,
            retry_count=retry_count,
        )
        return ExecutionResult(
            record=record,
            provider_reference=provider_reference,
            payment_link_url=payment_link_url,
            notification=notification,
            skipped_reason=skipped_reason,
        )


def _endpoint_for(action: ActionType) -> str:
    return {
        ActionType.RETRY_NOW: "orders.create",
        ActionType.RETRY_SCHEDULED: "orders.create",
        ActionType.RETRY_ALTERNATE_RAIL: "orders.create",
        ActionType.SEND_PAYMENT_LINK: "payment_link.create",
        ActionType.PRE_DEBIT_NUDGE: "notification.compose",
        ActionType.ESCALATE_HUMAN: "queue.enqueue",
    }.get(action, "unknown")
