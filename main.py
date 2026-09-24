"""
Runs the healthcare multi-agent system end to end: WhatsApp triage, patient
follow-up state tracking, clinical RAG queries, then plots telemetry.
"""

import os
import sys
from agents.multiagent_orchestrator import HealthcareMultiAgentOrchestrator
from visualizer.agent_telemetry_plots import plot_agent_telemetry_summary


def main():
    print("Healthcare Multi-Agent System & Clinical RAG")

    orchestrator = HealthcareMultiAgentOrchestrator()

    orchestrator.followup_agent.register_patient("PT-101", "David Miller", "Post-Op Knee Arthroscopy", discharge_day=0)
    orchestrator.followup_agent.register_patient("PT-102", "Elena Rostova", "Hypertension Management", discharge_day=0)
    print("Registered 2 patients for recovery follow-up monitoring.")

    simulation_events = [
        {"patient_id": "PT-201", "msg": "Hello, I am having sudden shortness of breath and severe chest pain.", "is_followup": False},
        {"patient_id": "PT-202", "msg": "My toddler has a high fever of 103F and is vomiting since morning.", "is_followup": False},
        {"patient_id": "PT-203", "msg": "Could you tell me what foods I should avoid with Type 2 Diabetes?", "is_followup": False},
        {"patient_id": "PT-204", "msg": "What are the common emergency warning signs for an acute asthma attack?", "is_followup": False},
        {"patient_id": "PT-205", "msg": "Can I reschedule my routine dental cleaning next week?", "is_followup": False},
        # Monitored Follow-up responses
        {"patient_id": "PT-101", "msg": "I am feeling worse, severe fever and increasing surgical pain around the joint.", "is_followup": True},
        {"patient_id": "PT-102", "msg": "Blood pressure readings are normal today, feeling much better!", "is_followup": True},
    ]

    for idx, event in enumerate(simulation_events, 1):
        print(f"\n--- Event {idx}: Inbound Message from [{event['patient_id']}] ---")
        print(f"Message: \"{event['msg']}\"")
        result = orchestrator.handle_incoming_message(
            patient_id=event["patient_id"],
            message=event["msg"],
            is_followup_response=event["is_followup"]
        )
        print(f"Routed Agent: [{result['routed_agent']}] | Latency: {result['latency_ms']:.2f} ms")
        print(f"Alert Status: {'ACTIVE ESCALATION' if result['alert'] else 'STABLE / ROUTINE'}")
        print(f"System Action: {result['response'].splitlines()[0]}")

    fig_path = plot_agent_telemetry_summary(
        orchestrator.telemetry_log,
        output_dir=os.path.join(os.path.dirname(__file__), "outputs", "figures")
    )
    print(f"\nSaved telemetry figure: {fig_path}")

    print(f"\nTotal Dispatches Processed : {len(orchestrator.telemetry_log)}")
    print(f"Average System Latency     : {sum(t['latency_ms'] for t in orchestrator.telemetry_log) / len(orchestrator.telemetry_log):.2f} ms")


if __name__ == "__main__":
    main()
