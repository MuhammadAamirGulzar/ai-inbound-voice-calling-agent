"""
Call control over Twilio's REST API: transfers and hangup.

Transfer design (warm-ish "whisper" transfer):
  1. The LLM emits transfer_call(reason, summary).
  2. The session speaks a short "transferring you now" line to the caller.
  3. We redirect the live call to <Dial> TwiML pointing at the staff
     number. The <Number url=...> whisper callback plays the AI's handoff
     summary to the staff member *before* the caller is bridged, so the
     human never answers blind.
  4. If nobody answers within the timeout, the caller hears an apology
     instead of dead air.

Redirecting the call makes Twilio tear down the media stream WebSocket,
which ends the CallSession naturally.

Uses httpx directly (async) instead of the sync twilio SDK so the event
loop is never blocked while holding dozens of live calls.
"""

from __future__ import annotations

import urllib.parse
from xml.sax.saxutils import escape

import httpx

TWILIO_API = "https://api.twilio.com/2010-04-01"


class TwilioCallControl:
    def __init__(self, config):
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.twilio_account_sid and self.config.twilio_auth_token)

    async def _update_call(self, call_sid: str, twiml: str) -> bool:
        url = (f"{TWILIO_API}/Accounts/{self.config.twilio_account_sid}"
               f"/Calls/{call_sid}.json")
        auth = (self.config.twilio_account_sid, self.config.twilio_auth_token)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, data={"Twiml": twiml}, auth=auth)
            if resp.status_code >= 300:
                print(f"[transfer] Twilio call update failed "
                      f"{resp.status_code}: {resp.text[:200]}")
                return False
            return True
        except Exception as e:
            print(f"[transfer] Twilio call update error: {e}")
            return False

    def build_transfer_twiml(self, target_number: str, summary: str = "") -> str:
        base = self.config.public_base_url
        number_attrs = ""
        if base and summary:
            whisper = f"{base}/twilio/transfer-whisper?summary=" + \
                      urllib.parse.quote(summary[:400])
            number_attrs = f' url="{escape(whisper, {chr(34): "&quot;"})}"'
        return (
            "<Response>"
            '<Dial timeout="25">'
            f"<Number{number_attrs}>{escape(target_number)}</Number>"
            "</Dial>"
            "<Say>Sorry, no one is available to take the call right now. "
            "Please call back later. Goodbye.</Say>"
            "</Response>"
        )

    async def transfer(self, call_sid: str, target_number: str,
                       summary: str = "") -> bool:
        if not self.enabled or not call_sid or not target_number:
            return False
        twiml = self.build_transfer_twiml(target_number, summary)
        return await self._update_call(call_sid, twiml)

    async def hangup(self, call_sid: str) -> bool:
        if not self.enabled or not call_sid:
            return False
        return await self._update_call(call_sid, "<Response><Hangup/></Response>")
