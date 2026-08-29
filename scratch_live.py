import uuid
from pathlib import Path

from recovery.execute.client import LiveClient, ProviderError, RecordingClient

ref = f"demo_{uuid.uuid4().hex[:8]}"
client = LiveClient.from_env()

# 1. Fresh link
r = client.create_payment_link(
    amount_paise=49900,
    reference_id=ref,
    description="Recovery agent demo - failed subscription",
    notes={"case_id": ref, "action": "send_payment_link"},
)
print("id:     ", r["id"])
print("status: ", r["status"])
print("url:    ", r["short_url"])

# 2. Same reference again - the provider must refuse
try:
    client.create_payment_link(
        amount_paise=49900,
        reference_id=ref,
        description="duplicate attempt",
        notes={"case_id": ref},
    )
    print("FAIL: duplicate was accepted")
except ProviderError as e:
    print("provider rejected duplicate:", e)
    print("retriable:", e.retriable)

# 3. Record fixtures with fresh references
rec = RecordingClient(client, Path("tests/fixtures/razorpay_live.json"))
rec.create_order(amount_paise=49900, receipt=f"rec_{uuid.uuid4().hex[:12]}", notes={"case_id": ref})
rec.create_payment_link(
    amount_paise=99900,
    reference_id=f"demo_{uuid.uuid4().hex[:8]}",
    description="Recovery demo 2",
    notes={"case_id": ref},
)
print("fixture written")
