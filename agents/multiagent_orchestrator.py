"""
Routes inbound patient messages to the WhatsApp triage, follow-up, or
clinical RAG agent depending on message type and follow-up state.
"""

import time
from typing import Dict, Any, Optional
from .whatsapp_triage_agent import WhatsAppTriageAgent
from .patient_followup_agent import PatientFollowUpAgent, FollowUpState
from .clinical_rag_agent import ClinicalRAGAgent


class HealthcareMultiAgentOrchestrator:
    """
    Central event-driven orchestrator coordinating specialized medical subagents.
    """

    def __init__(self, kb_path: Optional[str] = None):
        self.triage_agent = WhatsAppTriageAgent()
        self.followup_agent = PatientFollowUpAgent()
        self.rag_agent = ClinicalRAGAgent(kb_path=kb_path)
        self.telemetry_log = []

    def handle_incoming_message(
        self,
        patient_id: str,
        message: str,
        patient_name: Optional[str] = None,
        is_followup_response: bool = False
    ) -> Dict[str, Any]:
        """
        Routes incoming patient message through triage, RAG, or follow-up state handlers.
        """
        start_time = time.perf_counter()

        if is_followup_response and patient_id in self.followup_agent.patient_registry:
            result = self.followup_agent.process_patient_reply(patient_id, message)
            elapsed = (time.perf_counter() - start_time) * 1000

            telemetry_entry = {
                "patient_id": patient_id,
                "agent": "PatientFollowUpAgent",
                "action": "STATE_TRANSITION",
                "status": result["patient_state"].value,
                "latency_ms": elapsed
            }
            self.telemetry_log.append(telemetry_entry)

            return {
                "routed_agent": "PatientFollowUpAgent",
                "response": result["patient_reply"],
                "alert": result["alert_triggered"],
                "details": result,
                "latency_ms": elapsed
            }

        triage_eval = self.triage_agent.evaluate_triage(message)

        if triage_eval["triage_level"] in ["EMERGENCY", "URGENT"]:
            elapsed = (time.perf_counter() - start_time) * 1000
            telemetry_entry = {
                "patient_id": patient_id,
                "agent": "WhatsAppTriageAgent",
                "action": triage_eval["action"],
                "status": triage_eval["triage_level"],
                "latency_ms": elapsed
            }
            self.telemetry_log.append(telemetry_entry)

            return {
                "routed_agent": "WhatsAppTriageAgent",
                "response": triage_eval["reply_message"],
                "alert": triage_eval["requires_physician_call"],
                "details": triage_eval,
                "latency_ms": elapsed
            }

        rag_eval = self.rag_agent.answer_query(message)
        elapsed = (time.perf_counter() - start_time) * 1000

        telemetry_entry = {
            "patient_id": patient_id,
            "agent": "ClinicalRAGAgent",
            "action": "KNOWLEDGE_RETRIEVAL",
            "status": "GROUNDED_RESPONSE" if rag_eval["matched_disease"] else "FALLBACK",
            "latency_ms": elapsed
        }
        self.telemetry_log.append(telemetry_entry)

        combined_response = f"{rag_eval['response_text']}\n\n{rag_eval['disclaimer']}"

        return {
            "routed_agent": "ClinicalRAGAgent",
            "response": combined_response,
            "alert": False,
            "details": rag_eval,
            "latency_ms": elapsed
        }
