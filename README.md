# COMPSCI 767 LLM-Augmented Data Analysis Agent

This project is a small intelligent software agent prototype for COMPSCI 767. It takes a user goal and a CSV dataset, perceives the dataset state, plans analysis actions with an external OpenAI-compatible LLM API, validates the plan, executes safe local analysis tools, observes results, and updates future tool preferences from feedback.

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

- Streamlit interface for upload, goal input, trace inspection, charts, tables, and feedback.
- Need-specific visuals: correlation heatmap plus scatter, side-by-side group rankings, multi-metric trends with rolling average, and distribution plus boxplot views.
- Goal-aware column intent matching for choosing the most relevant metric, group, and date columns.
- Chart explanations in tool observations and final answers, so the user can interpret each visual without guessing.
- Deeper missing-value diagnostics with affected-row rates, severity labels, and grouped concentration checks.
- Large CSV row limiting so analysis stays responsive on oversized uploads.
- Auto header detection for CSV exports with preamble rows before the actual table.
- Date quality checks for mixed or messy date formats before trend analysis.
- Lightweight text analysis for feedback/comment datasets, including keywords and lexicon sentiment.
- Short conversation memory so follow-up prompts can reuse the previous goal and answer.
- Ambiguous goal detection with dataset-specific clarification suggestions.
- Stable CSV upload state handling that clears stale results when the file, header setting, or goal changes.
- OpenAI-compatible LLM configuration through environment variables.
- Deterministic fallback planner when no API key is available or the LLM fails.
- Goal-aware fallback planning so trend, group comparison, relationship, and quality prompts choose different tool paths.
- Safe predefined analysis tools; the LLM never executes arbitrary code.
- Safety guardrails that block requests to bypass validation, run arbitrary Python, execute shell commands, or inspect local files.
- Evaluation dashboard that runs deterministic edge-case checks and reports pass/fail results inside the app.
- Lightweight feedback memory that adjusts future tool priorities.
- Unit tests for perception, planning, tool execution, and feedback.

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

For CSV uploads, use `Auto detect header` when a file contains report title rows or notes before the actual table. The app caps analysis rows with `Max analysis rows` and reports when a large file is limited.

## Optional LLM Provider Configuration

The app uses an OpenAI-compatible `/chat/completions` interface.

You can either export variables in the terminal or create a local `.env` file. `.env` is ignored by Git so API keys are not committed.

```bash
export LLM_ENABLED=true
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openrouter/owl-alpha"
streamlit run app.py
```

Example provider settings:

```bash
# OpenRouter
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openrouter/owl-alpha"
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

The Streamlit app also includes an **Evaluation Dashboard**. Click `Run Evaluation Suite` to run the core grading-oriented scenarios with the LLM disabled: text-only data, numeric-only data, messy dates, complex-header CSV, missing values, empty CSV, no-header CSV, safety guardrail, unclear goal, and large trend data.

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

## Demo Video

Add the final two-minute demo link here after recording:

```text
Demo video: TODO
```

The demo should show: sample CSV, user goal, LLM or fallback plan, validated actions, charts/tables, final answer, and feedback score update.

## Report

The draft two-page Word report is in:

```text
report/COMPSCI767_Data_Analysis_Agent_Report.docx
```

Replace the GitHub and demo-video placeholders before final submission.

## Commit Checkpoints

Suggested local checkpoint history:

1. `Initial Streamlit data agent skeleton`
2. `Add dataset perception and state extraction`
3. `Add OpenAI-compatible LLM planner`
4. `Add safe analysis tools and validation`
5. `Expose ReAct-style trace and feedback memory`
6. `Add tests, README, report assets, and demo link`

## Safety Boundaries

- Uploaded CSV data is processed locally.
- The LLM receives a compact dataset profile, not arbitrary code execution rights.
- LLM tool selections are validated against an allowlist.
- Invalid LLM output falls back to deterministic planning.
- Unsafe goals are blocked before tool execution, so no charts or tables are generated for requests outside the CSV-analysis boundary.
