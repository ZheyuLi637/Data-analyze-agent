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
- Ambiguous goal detection with dataset-specific clarification suggestions.
- OpenAI-compatible LLM configuration through environment variables.
- Deterministic fallback planner when no API key is available or the LLM fails.
- Safe predefined analysis tools; the LLM never executes arbitrary code.
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
