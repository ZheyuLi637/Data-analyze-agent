from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from agent.agent_core import DataAnalysisAgent
from agent.clarification import clarification_context
from agent.llm_client import LLMConfig, OpenAICompatibleClient
from agent.memory import initial_tool_scores, update_tool_scores
from agent.perception import perceive_dataset


SAMPLE_DATA = Path("data/sample_sales.csv")
DEFAULT_GOAL = "Find useful patterns, risks, and next-step recommendations in this dataset."


st.set_page_config(page_title="COMPSCI 767 Data Analysis Agent", layout="wide")


def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA)


def ensure_state() -> None:
    if "tool_scores" not in st.session_state:
        st.session_state.tool_scores = initial_tool_scores()
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "goal_text" not in st.session_state:
        st.session_state.goal_text = DEFAULT_GOAL


def read_input_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_sample_data()


ensure_state()

st.title("COMPSCI 767 LLM-Augmented Data Analysis Agent")
st.caption("Perceive dataset state, plan with an external LLM API, execute safe tools, observe results, and learn from feedback.")

config = LLMConfig.from_env()
with st.sidebar:
    st.header("LLM Provider")
    st.write("OpenAI-compatible API configuration")
    st.code(
        "\n".join(
            [
                f"LLM_ENABLED={str(config.enabled).lower()}",
                f"LLM_BASE_URL={config.base_url}",
                f"LLM_MODEL={config.model}",
                f"LLM_API_KEY={'set' if config.api_key else 'not set'}",
            ]
        )
    )
    if config.ready:
        st.success("LLM planner enabled")
    else:
        st.warning("Fallback planner active")

    st.header("Feedback Scores")
    st.json(st.session_state.tool_scores)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
goal = st.text_area("Agent goal", height=90, key="goal_text")

df = read_input_data(uploaded)
preview_profile = perceive_dataset(df)
preview_clarification = clarification_context(goal, preview_profile)

if preview_profile.row_count == 0:
    st.error("Dataset has no rows. Upload a CSV with at least one data row before running analysis.")

if preview_clarification["ambiguous"]:
    suggestions = preview_clarification["suggestions"]
    if preview_clarification["requires_user_input"]:
        st.warning("The goal is not interpretable enough to choose safe actions. Choose a suggested focus before running the agent.")
    else:
        st.info("The goal is broad. Choose a suggested focus or run the agent with the first suggestion as planning context.")
    cols = st.columns(len(suggestions))
    for index, suggestion in enumerate(suggestions):
        with cols[index]:
            if st.button(suggestion, key=f"suggestion_{index}"):
                st.session_state.goal_text = suggestion
                st.rerun()

st.subheader("Dataset Preview")
st.dataframe(df.head(8), width="stretch")

if st.button("Run Agent", type="primary"):
    agent = DataAnalysisAgent(OpenAICompatibleClient(config))
    st.session_state.last_run = agent.run(df, goal, st.session_state.tool_scores)

run = st.session_state.last_run
if run:
    clarification = getattr(
        run,
        "clarification",
        {"ambiguous": False, "suggestions": [], "planning_goal": run.goal},
    )
    guardrail = getattr(run, "guardrail", {"blocked": False, "reason": "", "matched_terms": []})

    st.subheader("Agent Trace")
    trace_tabs = st.tabs(["Perceive", "Plan", "Act", "Observe", "Final Answer"])

    with trace_tabs[0]:
        st.json(run.profile.to_dict())

    with trace_tabs[1]:
        if guardrail["blocked"]:
            st.error("Safety guardrail blocked this request. No analysis tools were executed.")
            st.write(guardrail["reason"])
        if clarification.get("requires_user_input"):
            st.warning("The agent needs a clearer analysis goal before executing tools.")
        if clarification["ambiguous"]:
            st.info("Clarification suggestions were generated because the goal was broad.")
            st.write(clarification["suggestions"])
        st.write(f"Plan source: **{run.plan.source}**")
        if run.plan.error:
            st.warning(f"LLM planner fallback reason: {run.plan.error}")
        st.dataframe(
            pd.DataFrame([step.to_dict() for step in run.plan.steps]),
            width="stretch",
        )
        if run.plan.raw_response:
            with st.expander("Raw LLM planner response"):
                st.code(run.plan.raw_response)

    with trace_tabs[2]:
        for result in run.tool_results:
            st.markdown(f"**{result.title}**")
            st.write(result.observation)

    with trace_tabs[3]:
        st.json(run.trace)

    with trace_tabs[4]:
        st.write(run.final_answer)

    st.subheader("Tool Outputs")
    if not run.tool_results:
        st.info("No tools were executed for this run.")
    else:
        for result in run.tool_results:
            with st.expander(result.title, expanded=True):
                st.write(result.observation)
                if result.table is not None:
                    st.dataframe(result.table, width="stretch")
                if result.figure is not None:
                    st.pyplot(result.figure)

    if run.tool_results:
        st.subheader("Feedback")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Useful"):
                st.session_state.tool_scores = update_tool_scores(
                    st.session_state.tool_scores,
                    run.tools_used,
                    "useful",
                )
                st.rerun()
        with col2:
            if st.button("Not useful"):
                st.session_state.tool_scores = update_tool_scores(
                    st.session_state.tool_scores,
                    run.tools_used,
                    "not_useful",
                )
                st.rerun()
