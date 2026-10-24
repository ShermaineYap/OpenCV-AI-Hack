"""Amazon Bedrock narrator (optional, never in the safety-critical path).

The deterministic policy decides *what* to do. Bedrock (Amazon Nova via the
Converse API) only turns the structured visual evidence into a short, clear
message for the person cooking, and writes the end-of-session incident report.
If Bedrock is unavailable or slow, a template is used, so the loop never blocks.
Prompts contain only numbers and states - never images or personal data.
"""
from __future__ import annotations

import json
import os
import time

SYSTEM = ("You are BoilGuard, a kitchen safety assistant. Write in plain, calm English. "
          "Use the numbers given; never invent measurements. Keep messages under 35 words. "
          "Say what you saw, what you did, and what the cook should do next.")


def template_message(event: dict) -> str:
    kind = event.get("kind")
    eta = event.get("eta_s")
    eta_txt = f"~{eta:.0f}s" if isinstance(eta, (int, float)) and eta < 1e6 else "soon"
    fb = event.get("freeboard_mm")
    fb_txt = f"{fb:.0f} mm below the rim" if isinstance(fb, (int, float)) else ""
    if kind == "alert":
        return f"Foam is rising fast ({fb_txt}). Boil-over likely in {eta_txt}. Please check the stove."
    if kind == "intervene":
        return f"Boil-over predicted in {eta_txt}: I turned the heat down to LOW. Foam is {fb_txt}."
    if kind == "veto_window":
        return f"Boil-over predicted in {eta_txt}. Turning heat to LOW in {event.get('veto_s', 3):.0f}s unless you cancel."
    if kind == "escalate":
        return "Foam is still rising after lowering the heat, so I switched the burner OFF and sounded the alarm."
    if kind == "resolved":
        return "Foam is going down. Heat is on LOW. Restore heat from the app when you are ready."
    if kind == "low_visibility":
        return "Steam is blocking my view of the pot, so I turned on the hood fan and will act earlier to stay safe."
    if kind == "overflow":
        return "The pot boiled over. Burner switched OFF. Please wipe the hob before restarting."
    if kind == "confirm_rim":
        return "I could not find the pot rim confidently. Please confirm the pot outline in the app."
    if kind == "boiling":
        return "Boiling started. Watching the pot closely."
    return event.get("text", "BoilGuard update.")


class Narrator:
    def __init__(self, use_bedrock: bool | None = None, model_id: str | None = None,
                 region: str | None = None, max_calls: int = 4, timeout_s: float = 4.0):
        if use_bedrock is None:
            use_bedrock = os.environ.get("BOILGUARD_BEDROCK", "0") == "1"
        self.use = use_bedrock
        self.model_id = model_id or os.environ.get("BOILGUARD_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
        self.region = region or os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1"))
        self.max_calls = max_calls
        self.calls = 0
        self.log = []
        self._client = None
        self.timeout = timeout_s

    def _bedrock(self, prompt: str, max_tokens=160) -> str | None:
        if not self.use or self.calls >= self.max_calls:
            return None
        try:
            if self._client is None:
                import boto3
                from botocore.config import Config
                self._client = boto3.client("bedrock-runtime", region_name=self.region,
                                            config=Config(read_timeout=self.timeout, connect_timeout=2,
                                                          retries={"max_attempts": 1}))
            self.calls += 1
            t0 = time.time()
            r = self._client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2})
            text = r["output"]["message"]["content"][0]["text"].strip()
            self.log.append({"ok": True, "ms": round((time.time() - t0) * 1000), "model": self.model_id})
            return text
        except Exception as e:
            self.log.append({"ok": False, "error": type(e).__name__})
            return None

    def message(self, event: dict) -> tuple[str, str]:
        """Returns (text, source) where source is 'bedrock' or 'template'."""
        if event.get("kind") in ("alert", "intervene", "escalate", "overflow"):
            text = self._bedrock("Write the phone notification for this event:\n" + json.dumps(event))
            if text:
                return text, "bedrock"
        return template_message(event), "template"

    def incident_report(self, summary: dict, trace: list[dict]) -> tuple[str, str]:
        compact = [{k: e.get(k) for k in ("t", "state", "decision", "tools")} for e in trace if e.get("decision")][:30]
        prompt = ("Write a 4-6 sentence incident report for the cook from this BoilGuard session. "
                  "State whether a boil-over happened, the key prediction, the actions taken and when, "
                  "and one practical tip.\nSUMMARY: " + json.dumps(summary) + "\nDECISIONS: " + json.dumps(compact))
        text = self._bedrock(prompt, max_tokens=320)
        if text:
            return text, "bedrock"
        s = summary
        if s.get("overflowed"):
            out = f"The pot boiled over at {s.get('overflow_t', 0):.0f}s. "
        else:
            out = "No boil-over occurred. "
        acts = [e for e in trace if e.get("tools")]
        if acts:
            first = acts[0]
            out += f"First action at {first['t']:.0f}s ({first['decision']}). "
        out += f"{len(acts)} action(s) were taken in total. Tip: use a larger pot or lower heat for milk."
        return out, "template"
