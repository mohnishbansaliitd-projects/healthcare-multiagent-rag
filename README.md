# Healthcare Multi-Agent RAG

A small multi-agent system for patient messaging. An orchestrator routes each incoming message to one of three agents: a keyword-based triage agent, a patient follow-up state machine, or a retrieval-augmented clinical question-answering agent. The system is a software prototype that runs on a three-disease sample knowledge base. It is not a clinical tool.

## Motivation

Patient messages differ in urgency: some need immediate escalation, some belong to a monitored recovery pathway, and some are general questions that can be answered from verified protocols. This project explores separating these cases into distinct agents, with retrieval answers restricted to a curated knowledge base.

## Method

```
incoming message
   |-- follow-up reply from a registered patient --> PatientFollowUpAgent (state machine, may alert physician)
   |-- else WhatsAppTriageAgent (keyword rules)
   |        EMERGENCY or URGENT --> escalation reply, physician call flagged
   |        ROUTINE --> ClinicalRAGAgent
   '-- ClinicalRAGAgent: embed query (Ollama bge-m3), cosine similarity over KB chunks,
                         threshold 0.45, answer with Ollama granite3.1-dense:8b using only the retrieved context
                         (keyword matching if Ollama is unreachable) + disclaimer
```

Each dispatch is logged with agent, action, status and latency, and summarised in a telemetry figure.

## Dataset

`data/clinical_knowledge_base.json` holds 3 hand-written disease entries (type 2 diabetes, hypertension, acute upper respiratory infection) with symptoms, emergency flags, follow-up protocol and first-line management. No external dataset is used and the content has not been clinically reviewed.

## Results

Source: `outputs/figures/agent_telemetry_dashboard.png`, produced by `python main.py` on a 7-message simulation defined in `main.py`.

- Messages routed: WhatsAppTriageAgent 3, ClinicalRAGAgent 2, PatientFollowUpAgent 2.
- Mean routing latency printed on the figure: WhatsAppTriageAgent 0.06 ms, PatientFollowUpAgent 0.05 ms, ClinicalRAGAgent 21482.11 ms.

The RAG latency includes local LLM generation through Ollama. These are timings from one run on one machine, not an accuracy evaluation; no retrieval or triage accuracy has been measured.

## Repository structure

```
main.py                          simulation of 7 events and telemetry plot
agents/                          triage, follow-up, clinical RAG, orchestrator
data/clinical_knowledge_base.json
visualizer/agent_telemetry_plots.py
tests/test_healthcare_agents.py  unit tests
outputs/figures/                 telemetry dashboard
```

## How to run

No requirements file is included. From the imports: numpy, pandas, matplotlib, requests and pytest. For embedding-based RAG, install Ollama and pull the models:

```
pip install numpy pandas matplotlib requests pytest
ollama pull bge-m3
ollama pull granite3.1-dense:8b
python -m pytest tests
python main.py
```

Without Ollama the RAG agent prints a notice and falls back to keyword matching, so the code still runs. Embeddings are cached in `data/` after the first run.

## Limitations and next steps

- Triage uses fixed keyword lists, so it will miss paraphrases and can misclassify negations.
- The knowledge base has only 3 conditions and the similarity threshold was tuned on it by hand.
- No evaluation set; answer quality and triage accuracy are not measured.
- Not validated for clinical use and should not be used for real patient care.
- Next steps: add a labelled evaluation set, a larger reviewed knowledge base, and a requirements file.

Mohnish Bansal, B.Tech Materials Engineering, IIT Delhi
