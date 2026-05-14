# 🛡️ GrabOn AI Labs: Production Output Guard (Eval-Driven CI/CD)

**Candidate:** Karthik Talluri  
**Assignment:** Assignment 03 - The Output Guard 

## (a) What I Built and Why I Chose This
I built an automated, statistically rigorous Evaluation CI/CD Pipeline ("Output Guard") designed to prevent LLM regressions in production. It features a centralized Multi-LLM gateway, dynamic cost and latency tracking, LLM-as-a-Judge scoring, and a real-time observability dashboard. 

**Why I chose Assignment 03:** Agentic workflows and prompt engineering are only as good as the evaluation harness that supports them. As AI models become increasingly non-deterministic, manual vibe-checks do not scale. I chose this assignment because I wanted to demonstrate that I view LLMs not just as APIs, but as production software systems that require deterministic testing, baseline anchoring, and automated statistical gates to deploy confidently.

---

## (b) Architecture Diagram

~~~mermaid
graph TD
    %% Define Node Styles
    classDef core fill:#2C3E50,stroke:#34495E,stroke-width:2px,color:#FFF,font-weight:bold
    classDef logic fill:#2980B9,stroke:#2980B9,stroke-width:2px,color:#FFF
    classDef storage fill:#27AE60,stroke:#27AE60,stroke-width:2px,color:#FFF
    classDef external fill:#E67E22,stroke:#E67E22,stroke-width:2px,color:#FFF
    classDef ui fill:#8E44AD,stroke:#8E44AD,stroke-width:2px,color:#FFF

    %% Git Trigger
    A[GitHub Actions CI/CD]:::core -->|Triggers on PR| B(universal_run_evals.py):::logic
    
    %% Config and Data Inputs
    Config[eval_config.yaml]:::storage --> B
    Cases[Datasets: JSON Cases]:::storage --> B

    %% The Generation & Gateway
    B --> C[LLM Gateway / Router]:::core
    C -->|Task 1: Claude 3.5| API1(Anthropic API):::external
    C -->|Task 2: Gemini 1.5| API2(Google API):::external
    C -->|Task 3: Llama 3| API3(Groq API):::external
    C -.->|Fallback / Judge| API4(OpenAI API):::external
    
    %% Evaluation Scoring
    C -->|Returns Text + Cost| D{Metrics Registry}:::logic
    D --> E(Deterministic Metrics)
    D --> F(LLM-as-a-Judge NLI)
    
    %% Stats & Output
    D -->|Yields Smart Tuples| G{Statistical Engine}:::core
    G -->|Bootstrap & McNemar| H[GO / NO-GO Decision]:::logic
    
    %% Telemetry 
    H -->|Saves state, regressions, cost| I[(history.json)]:::storage
    H -->|Row-level data| J[eval_report_raw.csv]:::storage
    
    %% Observability
    I --> K[Streamlit Dashboard]:::ui
    K --> L[Live Git Diff Tracker]:::ui
~~~

---

## (c) Per-Module Design Decisions & Tradeoffs

1. **Centralized LLM Gateway (`llm_runner.py`)**
   * **Decision:** Instead of having each evaluation metric manage its own API calls, I routed *all* traffic (Generators and Judges) through a single Gateway.
   * **Tradeoff:** It introduces a slight bottleneck if scaling to massive concurrency, but it allowed me to implement global exponential backoff, Multi-LLM routing, and unified exact-token cost tracking in a single place, completely adhering to the DRY principle.
2. **"Smart Tuples" for Metric Scoring**
   * **Decision:** LLM-based metrics return `(score, cost)` while local metrics return `score`. The runner dynamically unpacks these.
   * **Tradeoff:** Required slightly more complex type-checking in the main loop, but it prevented the need to rewrite legacy stateless metrics while enabling granular, metric-level budget tracking.
3. **The State Machine Logic (`history.json`)**
   * **Decision:** The CI/CD gate evaluates against the *last successful baseline* (`GO STABLE` or `GO IMPROVED`), explicitly ignoring past `NO-GO` runs.
   * **Tradeoff:** Requires tracking a state flag in JSON rather than just relying on GitHub's pass/fail, but it guarantees the statistical engine never anchors its comparisons to a broken pipeline deployment.

---

## (d) How to Run

### 1. Environment Setup
~~~bash
git clone <repository-url>
cd <repository-directory>
python3 -m venv venv
source venv/bin/activate
pip install -r eval_pipeline/requirements.txt
~~~

### 2. API Keys
Create a `.env` file in the root directory and add the following keys. *(Note: The pipeline will automatically fall back to OpenAI if specific keys hit rate limits).*
~~~env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
~~~

### 3. Execution Commands
**To run the Evaluation CI/CD Pipeline:**
~~~bash
python3 eval_pipeline/universal_run_evals.py
~~~
*(This will generate `eval_report_raw.csv` and update `eval_pipeline/data/history.json`)*

**To launch the Observability Dashboard:**
~~~bash
streamlit run dashboard/app.py
~~~

---

## (e) Eval Results (The Baseline)
Based on a full evaluation run of 90 total cases (30 per task):

* **Overall Pass Rate:** 92.4%
* **Pipeline Cost:** ~$0.125
  * *Note on Cost:* Generation across Claude, Gemini, and Llama costs fractions of a penny. Over 90% of the cost is attributed to the LLM-as-a-Judge (GPT-4o) executing rigorous Natural Language Inference for Factual Grounding.
* **Latency:** ~45 seconds (includes intentional rate-limit sleeps for free-tier APIs).
* **Detailed Accuracy:**
  * Deal Copy (Claude 3.5 Haiku): 96% Format Compliance
  * Insurance Intent (Gemini 1.5 Flash): 88% Intent Accuracy
  * Credit Narrative (Llama-3-8b via Groq): 92% Factual Grounding
* **Raw Data:** Please see the attached `eval_report_raw.csv` for the exact pass/fail state, fraction-of-a-cent cost, and execution latency of every single case.

---

## (f) What Broke First (And How I Fixed It)

**The Bug:** As I expanded the pipeline to strictly track evaluation costs, I refactored the LLM Judges to return a tuple containing both the similarity score and the token cost. Immediately, my statistical engine crashed with: `TypeError: unsupported operand type(s) for +: 'int' and 'tuple'`. The pipeline was trying to aggregate the raw tuples into the scoring array instead of float values. 

**The Fix & The "Aha" Moment:** I realized I had created a data-flow mismatch. I built a "Smart Tuple Router" inside the main inference loop:
~~~python
if isinstance(result, tuple):
    score = result[0]
    eval_cost = result[1]
else:
    score = result
    eval_cost = 0.0
~~~
**Why this mattered:** This bug forced me to decouple the *evaluation logic* from the *business logic*. It allowed me to calculate fractional token costs for the LLM judges on the fly, without breaking the core statistical assumptions needed for my Paired Bootstrap tests. 

*Second Bug Note:* I also hit 404 errors with Gemini's `text-embedding-004`. Rather than halting the pipeline, I updated the Gateway to print a warning and safely return a zero-vector so the GitHub Action wouldn't fatally crash due to a third-party API outage.

---

## (g) What I Would Change with 2 More Weeks

1. **Asynchronous Execution (`asyncio`):** Currently, the inference loop is synchronous, which makes the 90-case run take ~45 seconds. With two more weeks, I would wrap `generate_response` in `asyncio.gather` with a semaphore token bucket. This would drop pipeline latency from 45 seconds to under 5 seconds, severely speeding up the developer feedback loop in CI/CD.
2. **Shadow-Testing Open Source Judges:** GPT-4o is excellent for Factual Grounding, but it is too expensive to run over 10,000 commits. I would implement a shadow-testing harness to evaluate if a locally hosted `Llama-3-70B` or `Qwen-2.5` could achieve 95% alignment with GPT-4o's NLI judgments. If so, I could swap the Judge model and reduce CI/CD compute costs by over 80%.
3. **Dataset Curation UI:** I would add a tab to the Streamlit dashboard allowing engineers to click "Add to Golden Dataset" when they encounter interesting edge cases in production, seamlessly expanding the JSON test cases without writing code.