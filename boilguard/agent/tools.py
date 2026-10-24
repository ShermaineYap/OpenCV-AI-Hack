"""Agent tools (actions) and the human channel.

Every tool call is logged by the agent with the visual evidence that caused it.
Backends:
* SimActuators  - drive the closed-loop simulator (evaluation, web demo)
* AwsActuators  - AWS IoT Core MQTT commands to a smart plug / cooktop / hood,
                  Amazon SNS notifications, S3 evidence snapshots
* LogActuators  - dry run (analysing an uploaded video: "would have done X")
"""
from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np


class Actuators:
    name = "base"

    def set_heat(self, level: float, reason: str) -> dict:
        return {"ok": True, "dry_run": True}

    def hood_fan(self, on: bool, reason: str) -> dict:
        return {"ok": True, "dry_run": True}

    def notify(self, severity: str, message: str, evidence: dict | None = None) -> dict:
        return {"ok": True, "dry_run": True}

    def alarm(self, on: bool) -> dict:
        return {"ok": True, "dry_run": True}


class LogActuators(Actuators):
    name = "dry-run"


class SimActuators(Actuators):
    name = "simulator"

    def __init__(self, sim):
        self.sim = sim

    def set_heat(self, level, reason):
        self.sim.set_power(level, "agent")
        return {"ok": True, "heat": level}

    def hood_fan(self, on, reason):
        self.sim.set_fan(on, "agent")
        return {"ok": True, "fan": on}

    def notify(self, severity, message, evidence=None):
        return {"ok": True, "delivered": "sim-user"}

    def alarm(self, on):
        return {"ok": True, "alarm": on}


class AwsActuators(Actuators):
    """Publishes device commands to AWS IoT Core and alerts to Amazon SNS.

    Topic layout: ``boilguard/<device_id>/cmd`` with JSON payloads such as
    ``{"cmd": "set_heat", "level": 0.35, "reason": "...", "request_id": "..."}``.
    A Tasmota/Shelly smart plug, an IoT-enabled induction hob or an ESP32 relay can
    subscribe to it. Calls are rate limited so a public demo cannot spam devices.
    """
    name = "aws"

    def __init__(self, device_id="demo-stove", region=None, sns_topic_arn=None,
                 max_msgs=12, session_id=None):
        import boto3
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.device_id = device_id
        self.topic = f"boilguard/{device_id}/cmd"
        self.sns_arn = sns_topic_arn or os.environ.get("BOILGUARD_SNS_TOPIC")
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.budget = max_msgs
        self._iot = None
        self._sns = boto3.client("sns", region_name=self.region) if self.sns_arn else None
        self._boto3 = boto3

    def _iot_client(self):
        if self._iot is None:
            ep = os.environ.get("BOILGUARD_IOT_ENDPOINT")
            if not ep:
                ep = self._boto3.client("iot", region_name=self.region).describe_endpoint(
                    endpointType="iot:Data-ATS")["endpointAddress"]
            self._iot = self._boto3.client("iot-data", region_name=self.region, endpoint_url=f"https://{ep}")
        return self._iot

    def _publish(self, payload: dict) -> dict:
        if self.budget <= 0:
            return {"ok": False, "error": "rate-limited"}
        self.budget -= 1
        payload = {**payload, "device": self.device_id, "session": self.session_id, "ts": time.time()}
        try:
            self._iot_client().publish(topic=self.topic, qos=1, payload=json.dumps(payload).encode())
            return {"ok": True, "topic": self.topic}
        except Exception as e:  # never let cloud errors break the safety loop
            return {"ok": False, "error": type(e).__name__}

    def set_heat(self, level, reason):
        return self._publish({"cmd": "set_heat", "level": level, "reason": reason})

    def hood_fan(self, on, reason):
        return self._publish({"cmd": "hood_fan", "on": on, "reason": reason})

    def alarm(self, on):
        return self._publish({"cmd": "alarm", "on": on})

    def notify(self, severity, message, evidence=None):
        out = self._publish({"cmd": "notify", "severity": severity, "message": message})
        if self._sns and severity in ("warning", "critical") and self.budget > 0:
            try:
                self._sns.publish(TopicArn=self.sns_arn, Subject=f"BoilGuard {severity}", Message=message[:1000])
                out["sns"] = True
            except Exception as e:
                out["sns_error"] = type(e).__name__
        return out


class MultiActuators(Actuators):
    """Fan-out: e.g. drive the simulator AND mirror commands to AWS IoT."""

    def __init__(self, *backends):
        self.backends = backends
        self.name = "+".join(b.name for b in backends)

    def _all(self, fn, *a):
        return {b.name: getattr(b, fn)(*a) for b in self.backends}

    def set_heat(self, level, reason):
        return self._all("set_heat", level, reason)

    def hood_fan(self, on, reason):
        return self._all("hood_fan", on, reason)

    def notify(self, severity, message, evidence=None):
        return self._all("notify", severity, message, evidence)

    def alarm(self, on):
        return self._all("alarm", on)


# --------------------------------------------------------------- human channel
class HumanChannel:
    """Interface to the person cooking (phone notification with buttons)."""

    def alert(self, t: float, message: str):
        pass

    def request_approval(self, t: float, action: str, deadline_s: float) -> str:
        return "req"

    def poll(self, t: float):
        """Returns list of events: {'type': 'ack'|'approve'|'deny'|'veto'|'took_over', ...}"""
        return []


class NoHuman(HumanChannel):
    pass


class SimHuman(HumanChannel):
    """Simulated cook. If present, reacts to alerts after a reaction time by
    turning the heat down themselves (taking over); answers approval requests."""

    def __init__(self, sim, present: bool, reaction_s: float, approve: bool = True):
        self.sim, self.present, self.reaction = sim, present, reaction_s
        self.approve = approve
        self.pending = []    # (due_t, event)
        self.took_over = False

    def alert(self, t, message):
        if self.present and not self.took_over and not any(e["type"] == "took_over" for _, e in self.pending):
            self.pending.append((t + self.reaction, {"type": "took_over"}))

    def request_approval(self, t, action, deadline_s):
        rid = f"req-{len(self.pending)}"
        if self.present:
            self.pending.append((t + self.reaction, {"type": "approve" if self.approve else "deny",
                                                     "id": rid, "action": action}))
        return rid

    def poll(self, t):
        due = [e for (d, e) in self.pending if d <= t]
        self.pending = [(d, e) for (d, e) in self.pending if d > t]
        for e in due:
            if e["type"] == "took_over":
                self.took_over = True
                if self.sim.s.P > 0.3:
                    self.sim.set_power(0.3, "human")
        return due


def sample_human(rng: np.random.Generator, sim, p_present=0.6):
    present = bool(rng.random() < p_present)
    reaction = float(np.clip(rng.lognormal(np.log(6.0), 0.5), 2.0, 25.0))
    return SimHuman(sim, present, reaction)
