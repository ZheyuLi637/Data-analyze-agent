# COMPSCI 767 LLM-Augmented Data Analysis Agent

This project is a small intelligent software agent prototype for COMPSCI 767. It takes a user goal and a CSV dataset, perceives the dataset state, plans analysis actions with an external OpenAI-compatible LLM API, validates the plan, executes safe local analysis tools, observes results, and updates future tool preferences from feedback.

## Demo Video

[Watch the 2-minute demo video](https://docs.google.com/videos/d/1m1gszfe3s03AKiceb-txTS5ZC2ppyNMh3z9RxY42tXg/edit?usp=drive_link)

The demo shows the working agent loop: CSV upload, user goal, LLM or fallback planning, validated tool execution, charts/tables, final answer, trace inspection, and feedback memory.

## Why this is an agent, not just a chatbot

The system does not only respond with text. It follows an agent loop:

```text
User goal + CSV
  -> perceive dataset state
  -> plan actions with an LLM-compatible planner
  -> validate selected tools
  -> act through safe pandas/matplotlib tools
  -> observe tool outputs
  -> synthesize a final answer
  -> update feedback scores
```

This maps to course concepts: environment, observation/state, action, policy, tool use, ReAct-style trace, memory, reward-like feedback, evaluation, and guardrails.

## Features

### 1. Core agent workflow

- Upload a CSV file and enter a user analysis goal.
- Inspect the final answer, tool outputs, charts, tables, and feedback controls.
- View the full agent trace: perceive, plan, act, and observe.

### 2. Dataset perception

- Detect numeric, categorical, date, and text columns.
- Detect missing values, messy dates, high-cardinality columns, and empty datasets.
- Handle messy CSV headers, no-header CSV files, large CSV files, and numeric-looking strings such as `$99.5` or `35%`.

### 3. Analysis tools

- Generate dataset summaries, missing-value checks, correlations, group comparisons, trends, charts, text analysis, statistical tests, and predictive baselines.
- Use goal-aware column intent matching to choose the most relevant metric, group, and date columns.
- Use bounded statistical methods such as Bonferroni-adjusted Pearson checks, lightweight ANOVA, confidence interval labels, and train/test predictive baselines.

### 4. Charts and explanations

- Generate need-specific visuals such as correlation heatmaps, scatter plots, group ranking charts, trend charts, distribution plots, and boxplots.
- Explain charts in tool observations and final answers so the user can interpret the visualization.

### 5. LLM planning and fallback

- Use an OpenAI-compatible LLM provider to help choose analysis actions.
- Keep execution local and safe; the LLM can select tools but cannot run arbitrary code.
- Fall back to deterministic planning when no API key is available, the model fails, or the output is invalid.

### 6. Safety and guardrails

- Validate every LLM-selected tool against an allowlist.
- Block prompt injection, arbitrary Python execution, shell command requests, local file inspection, and unsupported tool requests.
- Prevent misleading analysis by skipping unsuitable tools for text-only, numeric-only, messy-date, high-cardinality, and small-sample datasets.

### 7. Memory and feedback

- Store lightweight conversation memory for follow-up questions.
- Update tool scores from useful or not useful feedback.
- Detect ambiguous goals and suggest clearer dataset-specific analysis goals.

### 8. Evaluation

- Provide an Evaluation Dashboard with deterministic edge-case checks.
- Include unit tests for perception, planning, tool execution, guardrails, feedback, and evaluation behavior.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app works without an LLM key by using the fallback planner.

If the goal is broad, for example `analyze this data`, the app suggests clearer analysis goals based on the observed dataset columns. The user can click one suggestion, or run the agent directly; in that case the first suggestion is added as planning context and shown in the trace.

For CSV uploads, use `Auto detect header` when a file contains report title rows or notes before the actual table. The app caps analysis rows internally to keep large files responsive.

## Optional LLM Provider Configuration

The app uses an OpenAI-compatible `/chat/completions` interface.

You can either export variables in the terminal or create a local `.env` file. `.env` is ignored by Git so API keys are not committed.

```bash
export LLM_ENABLED=true
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="openai/gpt-oss-120b"
streamlit run app.py
```

Example provider settings:

```bash
# Groq
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="openai/gpt-oss-120b"

# Cerebras
export LLM_BASE_URL="https://api.cerebras.ai/v1"
export LLM_MODEL="gpt-oss-120b"

# OpenRouter
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="deepseek/deepseek-v4-flash:free"
export LLM_APP_NAME="COMPSCI 767 Data Analysis Agent"
export LLM_SITE_URL="http://localhost:8502"

# DeepSeek
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"

# OpenAI
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"

# Qwen compatible mode
export LLM_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"
```

## Test

```bash
python -m unittest discover tests
```

The Streamlit app also includes an **Evaluation Dashboard**. Click `Run Evaluation Suite` to run the core grading-oriented scenarios with the LLM disabled: text-only data, numeric-only data, messy dates, complex-header CSV, missing values, empty CSV, no-header CSV, safety guardrail, unclear goal, large trend data, topic modeling, statistical testing, predictive modeling, causal guardrails, high-cardinality groups, numeric-string inference, prompt injection, and train/test prediction.

## Boundary CSV Fixtures

The `data/` folder includes datasets for manual edge-case testing:

- `edge_text_only.csv`: one text column; should skip correlation and trend analysis.
- `edge_numeric_only.csv`: numeric-only data; should skip group comparison.
- `edge_messy_dates.csv`: mixed date quality; should run trend analysis only if dates are reliably detected.
- `edge_complex_header.csv`: preamble rows before the actual header; use auto header detection.
- `edge_many_missing.csv`: many missing values; should prioritize missing value checking.
- `edge_empty_header_only.csv`: header-only CSV; should show `Dataset has no rows`.
- `edge_no_header.csv`: no header row; turn off `First row contains headers` before running.
- `edge_large_sales.csv`: larger sample for checking load time and chart responsiveness.
- `edge_high_cardinality.csv`: high-cardinality ID grouping; should skip unreadable group comparison output.
- `edge_numeric_strings.csv`: currency and percent strings; should infer numeric columns for analysis.

## Commit Checkpoints

Brief design evolution recorded in the repository history:

1. `c8ae0c9 Initial data agent project skeleton`: created the Streamlit prototype and basic agent run path.
2. `0b8cbe4 Add dataset perception and state extraction`: added dataset profiling for rows, columns, numeric/categorical/date/text fields, and missing values.
3. `fcaf9c2 Add OpenAI-compatible LLM planner`: added provider-neutral LLM planning through environment variables.
4. `4f69eac Add safe analysis tools and chart generation`: added allowlisted pandas/matplotlib tools so the LLM cannot execute arbitrary code.
5. `aebdebd Expose agent trace and feedback memory`: made the perceive-plan-act-observe loop visible and added reward-like feedback.
6. `9d17cf2 Add safety guardrails and goal-aware fallback planning`: added prompt safety checks and deterministic behavior when no API key is available.
7. `ca62492 Add advanced need-specific charts`: improved visuals so charts match the analysis goal instead of using one generic chart type.
8. `4a94841 Add evaluation dashboard`: added deterministic edge-case testing inside the app.
9. `225803e Harden analysis tools and evaluation`: improved reliability for high-cardinality groups, statistics, text analysis, and error messages.
10. `9b7c0fa Tighten planner for edge datasets`: improved tool selection for no-header, numeric-only, high-cardinality, and small-sample datasets.

## Repository Contents

- `app.py`: Streamlit interface and user workflow.
- `agent/`: perception, planning, LLM client, tool execution, memory, and core agent loop.
- `data/`: sample and edge-case CSV files for manual testing.
- `tests/`: unit tests for agent decisions, tools, guardrails, and evaluation behavior.
- `.env.example`: safe template for optional LLM provider configuration.

## Future Work

- Add richer multi-turn reasoning so the agent can decompose longer analysis goals across several dependent steps.
- Add stronger statistical modeling with clearer assumptions, model diagnostics, and uncertainty reporting.
- Extend text analysis with embeddings or topic modeling libraries for larger feedback datasets.
- Add optional export of the final answer, trace, tables, and charts into a report-ready PDF.
- Add a hosted deployment option so graders can run the app without local setup.

## Safety Boundaries

- Uploaded CSV data is processed locally.
- The LLM receives a compact dataset profile, not arbitrary code execution rights.
- LLM tool selections are validated against an allowlist.
- Invalid LLM output falls back to deterministic planning.
- Unsafe goals are blocked before tool execution, so no charts or tables are generated for requests outside the CSV-analysis boundary.
