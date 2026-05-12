# 🛡️ GrabOn AI Output Guard
**Production Evaluation Framework for Agentic LLMs**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Executive Summary
Large Language Models operate stochastically, meaning traditional deterministic CI/CD assertions (`assert output == expected`) fail in AI-native applications. For a high-scale environment like GrabOn, a "silent degradation" (e.g., an LLM hallucinating a 38% GMV growth metric in a credit narrative) introduces severe regulatory and operational risks.

The **AI Output Guard** is a strictly automated, statistical evaluation pipeline. It intercepts prompt or model changes, evaluates them against an immutable dataset of 90 high-variance edge cases, calculates exact statistical significance (McNemar's & Paired Bootstrap), and explicitly blocks GitHub Pull Requests if a regression is detected.

## 🏗️ System Architecture

1. **Immutable Ground Truth (`data/cases.json`):** 90 edge-case scenarios covering Deal Copy (WhatsApp limits), Credit Underwriting (Factual Grounding), and Insurance (Ambiguity/ECE).
2. **Git-Native Prompt Versioning (`prompt_manager.py`):** Utilizes `hashlib` to generate SHA-256 content-addressable hashes, ensuring production prompts are strictly immutable.
3. **Multi-Model Scoring Engine (`scoring/`):** deterministic math, Vector Embeddings (Gemini), and LLM-as-a-Judge (OpenAI) to evaluate qualitative outputs.
4. **Statistical Decision Gate (`statistical_engine.py`):** Differentiates random stochastic noise from actual regressions using 95% Confidence Intervals.
5. **Telemetry Observability (`app.py`):** Streamlit dashboard for real-time drift tracking and anomaly attribution.

## 📂 Directory Structure
```text
GrabOn-AI-Guard/
├── .github/workflows/
│   └── eval_gate.yml           # GitHub Actions CI/CD configuration
├── eval_pipeline/
│   ├── data_creation.py        # Generates the 90 edge-case JSONs
│   ├── run_evals.py            # Main Orchestrator & PR Blocker
│   ├── prompt_manager.py       # Hashing and Versioning Engine
│   ├── stats/
│   │   └── statistical_engine.py # McNemar & Bootstrap (scipy)
│   └── scoring/
│       ├── format_compliance.py      # Deterministic step-decay scoring
│       ├── semantic_similarity.py    # Gemini text-embedding-004
│       ├── llm_judge.py              # OpenAI gpt-4o-mini (NLI / Quality)
│       └── confidence_calibration.py # Expected Calibration Error (ECE)
├── data/
│   └── history.json            # Local JSON database for telemetry
├── app.py                      # Streamlit Dashboard UI
└── requirements.txt            # Project dependencies