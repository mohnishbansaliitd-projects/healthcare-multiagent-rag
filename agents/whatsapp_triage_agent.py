"""
WhatsApp Patient Intake and Urgent Triage Autonomous Agent.

Classifies incoming patient messages into triage urgency tiers:
- EMERGENCY (immediate 911/ER redirection)
- URGENT (same-day clinic consultation)
- ROUTINE (standard appointment booking or general inquiry)
"""

import re
from typing import Dict, Any, Tuple


class WhatsAppTriageAgent:
    """
    Evaluates patient incoming chat messages and assigns triage classification.
    """

    EMERGENCY_KEYWORDS = [
        "chest pain", "shortness of breath", "severe bleeding", "sudden numbness",
        "loss of consciousness", "stroke", "paralysis", "anaphylaxis", "cyanosis",
        "unconscious", "cannot breathe"
    ]

    URGENT_KEYWORDS = [
        "high fever", "vomiting blood", "severe pain", "asthma attack", "worsening",
        "dizziness", "spreading rash", "glucose 350", "bp 180"
    ]

    def evaluate_triage(self, patient_message: str) -> Dict[str, Any]:
        """Classifies triage level and constructs conversational response."""
        msg_lower = patient_message.lower()

        for kw in self.EMERGENCY_KEYWORDS:
            if kw in msg_lower:
                return {
                    "triage_level": "EMERGENCY",
                    "flagged_keyword": kw,
                    "action": "IMMEDIATE_ER_ESCALATION",
                    "reply_message": "CRITICAL ALERT: Your symptoms may indicate a medical emergency. "
                                     "Please call emergency services (911/112) or go to the nearest emergency department immediately.",
                    "requires_physician_call": True
                }

        for kw in self.URGENT_KEYWORDS:
            if kw in msg_lower:
                return {
                    "triage_level": "URGENT",
                    "flagged_keyword": kw,
                    "action": "SAME_DAY_PHYSICIAN_SLOT",
                    "reply_message": "We have flagged your condition as URGENT. A same-day priority appointment is being reserved, "
                                     "and our on-call nurse has been notified.",
                    "requires_physician_call": True
                }

        return {
            "triage_level": "ROUTINE",
            "flagged_keyword": None,
            "action": "STANDARD_SCHEDULING",
            "reply_message": "Thank you for reaching out. We have logged your request and can schedule your next routine appointment.",
            "requires_physician_call": False
        }
