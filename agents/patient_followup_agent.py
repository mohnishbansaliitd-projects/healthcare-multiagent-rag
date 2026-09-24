"""
Autonomous Patient Follow-Up and Post-Consultation Monitoring Agent.

Manages state-machine tracking of recovery milestones, automated check-in messages,
and automatic escalation to clinical staff if symptoms worsen.
"""

from typing import Dict, Any, List, Optional
from enum import Enum


class FollowUpState(str, Enum):
    SCHEDULED = "SCHEDULED"
    CHECKIN_SENT = "CHECKIN_SENT"
    PATIENT_RESPONDED = "PATIENT_RESPONDED"
    SYMPTOMS_RESOLVED = "SYMPTOMS_RESOLVED"
    PHYSICIAN_ALERT_TRIGGERED = "PHYSICIAN_ALERT_TRIGGERED"


class PatientFollowUpAgent:
    """
    State-machine tracking for chronic disease & post-discharge patients.
    """

    def __init__(self):
        self.patient_registry: Dict[str, Dict[str, Any]] = {}

    def register_patient(
        self,
        patient_id: str,
        name: str,
        condition: str,
        discharge_day: int = 0
    ) -> Dict[str, Any]:
        """Registers a patient for autonomous recovery tracking."""
        record = {
            "patient_id": patient_id,
            "name": name,
            "condition": condition,
            "state": FollowUpState.SCHEDULED,
            "discharge_day": discharge_day,
            "history": []
        }
        self.patient_registry[patient_id] = record
        return record

    def dispatch_scheduled_checkin(self, patient_id: str, current_day: int) -> Optional[str]:
        """Generates proactive automated check-in prompt based on condition protocol."""
        if patient_id not in self.patient_registry:
            return None
        record = self.patient_registry[patient_id]

        record["state"] = FollowUpState.CHECKIN_SENT
        msg = f"Hello {record['name']}, this is your healthcare assistant. As part of your {record['condition']} recovery plan (Day {current_day}), " \
              f"how are your symptoms today? Please reply 'Better', 'Same', or 'Worse' along with any new sensations."
        record["history"].append({"day": current_day, "event": "CHECKIN_SENT", "message": msg})
        return msg

    def process_patient_reply(self, patient_id: str, reply_text: str) -> Dict[str, Any]:
        """Processes patient feedback and executes state transition."""
        if patient_id not in self.patient_registry:
            raise ValueError(f"Patient {patient_id} not found")

        record = self.patient_registry[patient_id]
        reply_lower = reply_text.lower()

        if "worse" in reply_lower or "severe" in reply_lower or "fever" in reply_lower or "pain" in reply_lower:
            record["state"] = FollowUpState.PHYSICIAN_ALERT_TRIGGERED
            alert_msg = f"CLINICAL ESCALATION: Patient {record['name']} ({patient_id}) reported worsening symptoms: '{reply_text}'"
            record["history"].append({"event": "ALERT_TRIGGERED", "details": alert_msg})
            return {
                "patient_state": record["state"],
                "alert_triggered": True,
                "physician_notification": alert_msg,
                "patient_reply": "We noticed you are feeling worse. We have immediately notified your primary physician who will call you shortly."
            }
        else:
            record["state"] = FollowUpState.SYMPTOMS_RESOLVED
            record["history"].append({"event": "RESOLVED", "details": reply_text})
            return {
                "patient_state": record["state"],
                "alert_triggered": False,
                "physician_notification": None,
                "patient_reply": "Thank you for the update! Glad to hear your recovery is progressing well. We will check in again next week."
            }
