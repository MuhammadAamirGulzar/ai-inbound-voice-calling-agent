"""
Twilio webhook signature validation (X-Twilio-Signature).

Twilio signs every webhook request with HMAC-SHA1 over the exact URL it
requested plus the sorted POST parameters, keyed by the account's auth
token. Validating it means only Twilio can trigger call handling — an
attacker who discovers the /voice URL can't open media streams or burn
LLM/TTS spend.

Implemented on the stdlib (same algorithm as twilio.request_validator)
so it stays importable in test environments without the twilio SDK.

Enable with TWILIO_VALIDATE_SIGNATURE=1. Behind a proxy/tunnel the URL
Twilio signed is the public one, so PUBLIC_BASE_URL must be set for the
reconstruction to match.
"""

from __future__ import annotations

import base64
import hashlib
import hmac


def compute_twilio_signature(auth_token: str, url: str, params: dict) -> str:
    payload = url + "".join(k + str(params[k]) for k in sorted(params))
    digest = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"),
                      hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def twilio_signature_valid(auth_token: str, url: str, params: dict,
                           signature: str) -> bool:
    if not auth_token or not signature:
        return False
    expected = compute_twilio_signature(auth_token, url, params)
    return hmac.compare_digest(expected, signature)
