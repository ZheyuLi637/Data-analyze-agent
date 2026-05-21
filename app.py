from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from agent.agent_core import DataAnalysisAgent
from agent.clarification import clarification_context
from agent.csv_loader import LoadedCSV, load_csv_bytes, load_csv_path
from agent.evaluation import run_evaluation
from agent.llm_client import LLMConfig, OpenAICompatibleClient
from agent.memory import initial_tool_scores, update_tool_scores
from agent.perception import perceive_dataset


SAMPLE_DATA = Path("data/sample_sales.csv")
DEFAULT_GOAL = "Find useful patterns, risks, and next-step recommendations in this dataset."


st.set_page_config(page_title="COMPSCI 767 Data Analysis Agent", layout="wide")


def ensure_state() -> None:
    if "tool_scores" not in st.session_state:
        st.session_state.tool_scores = initial_tool_scores()
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "goal_text" not in st.session_state:
        st.session_state.goal_text = DEFAULT_GOAL
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "active_dataset_id" not in st.session_state:
        st.session_state.active_dataset_id = None
    if "last_run_dataset_id" not in st.session_state:
        st.session_state.last_run_dataset_id = None
    if "active_request_id" not in st.session_state:
        st.session_state.active_request_id = None
    if "last_run_request_id" not in st.session_state:
        st.session_state.last_run_request_id = None
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "pending_goal_text" in st.session_state:
        st.session_state.goal_text = st.session_state.pop("pending_goal_text")


def read_input_data(uploaded_file, header_mode: str, max_rows: int) -> LoadedCSV:
    if uploaded_file is not None:
        return load_csv_bytes(uploaded_file.getvalue(), uploaded_file.name, header_mode, max_rows=max_rows)
    return load_csv_path(SAMPLE_DATA, "present", max_rows=max_rows)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(45, 125, 141, 0.12), transparent 28rem),
                linear-gradient(180deg, #f7f9fc 0%, #eef3f6 100%);
            color: #17202a;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #dfe6ec;
        }
        .hero {
            border: 1px solid #dfe8ed;
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            background: linear-gradient(135deg, #ffffff 0%, #edf7f5 58%, #f8fbff 100%);
            box-shadow: 0 16px 45px rgba(34, 52, 68, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.15rem;
            line-height: 1.1;
            margin: 0 0 0.45rem 0;
            letter-spacing: 0;
        }
        .hero p {
            color: #536471;
            margin: 0;
            font-size: 1rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }
        .pill {
            border: 1px solid #cfe2df;
            background: #f5fbfa;
            color: #1d5f67;
            padding: 0.28rem 0.58rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .section-label {
            color: #58706f;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }
        .soft-note {
            color: #60717e;
            font-size: 0.9rem;
            margin-top: -0.25rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dfe8ed;
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            box-shadow: 0 10px 28px rgba(34, 52, 68, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: #5b6b76;
            font-weight: 650;
        }
        .stButton > button {
            border-radius: 9px;
            font-weight: 700;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🤖 Data Insight Agent</h1>
            <p>LLM-augmented CSV analysis with tool validation, ReAct-style tracing, feedback memory, and evaluation.</p>
            <div class="pill-row">
                <span class="pill">Perceive</span>
                <span class="pill">Plan</span>
                <span class="pill">Act with tools</span>
                <span class="pill">Observe</span>
                <span class="pill">Learn from feedback</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


ensure_state()
inject_styles()

render_hero()

config = LLMConfig.from_env()
with st.sidebar:
    st.header("Task Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    header_mode = "present"
    if uploaded is not None:
        header_choice = st.selectbox(
            "CSV header handling",
            ["Auto detect header", "First row contains headers", "No header row"],
            index=0,
            help="Use auto detect for CSV files with report notes or blank lines before the actual table.",
        )
        header_mode = {
            "Auto detect header": "auto",
            "First row contains headers": "present",
            "No header row": "absent",
        }[header_choice]
    max_analysis_rows = st.number_input(
        "Max analysis rows",
        min_value=100,
        max_value=100000,
        value=5000,
        step=1000,
        help="Large CSVs are capped to keep analysis responsive. The app reports when it samples the first rows.",
    )

    st.divider()
    st.header("LLM Provider")
    if config.ready:
        st.success("LLM planner enabled")
    else:
        st.warning("Fallback planner active")
    with st.expander("OpenAI-compatible config", expanded=False):
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

    st.divider()
    st.header("Memory")
    st.metric("Prior turns", len(st.session_state.conversation_memory))
    if st.button("Clear Memory"):
        st.session_state.conversation_memory = []
        st.rerun()
    with st.expander("Feedback tool scores", expanded=False):
        st.json(st.session_state.tool_scores)

section_label("Task Configuration")
task_col, context_col = st.columns([2.1, 1])
with task_col:
    with st.container(border=True):
        st.markdown("### Agent Goal")
        goal = st.text_area(
            "Agent goal",
            height=118,
            key="goal_text",
            label_visibility="collapsed",
            placeholder="Example: Compare profit by customer segment and explain the strongest group.",
        )
        st.markdown(
            '<div class="soft-note">Describe the decision you want the agent to support. Follow-up prompts can reuse recent memory.</div>',
            unsafe_allow_html=True,
        )

loaded_csv = read_input_data(uploaded, header_mode, int(max_analysis_rows))
current_request_id = (loaded_csv.source_id, goal)
if current_request_id != st.session_state.active_request_id:
    st.session_state.active_dataset_id = loaded_csv.source_id
    st.session_state.active_request_id = current_request_id
    st.session_state.last_run = None
    st.session_state.last_run_dataset_id = None
    st.session_state.last_run_request_id = None

df = None
preview_profile = None
preview_clarification = None
if not loaded_csv.error:
    df = loaded_csv.dataframe
    assert df is not None
    preview_profile = perceive_dataset(df)
    preview_clarification = clarification_context(goal, preview_profile)

with context_col:
    with st.container(border=True):
        st.markdown("### Runtime Context")
        st.metric("Memory turns", len(st.session_state.conversation_memory))
        st.metric("LLM mode", "Enabled" if config.ready else "Fallback")
        if loaded_csv.error:
            st.error("CSV not ready")
        else:
            assert preview_profile is not None
            st.caption(loaded_csv.source_name)
            st.caption(f"{preview_profile.row_count} loaded rows / {loaded_csv.original_row_count} source rows")

if loaded_csv.error:
    st.error(loaded_csv.error)
    st.info("No analysis will run until the uploaded CSV can be read.")
else:
    assert df is not None
    assert preview_profile is not None
    assert preview_clarification is not None

    for note in loaded_csv.notes or []:
        st.info(note)

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
                if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                    st.session_state.pending_goal_text = suggestion
                    st.rerun()

    section_label("Dataset Overview")
    missing_total = sum(preview_profile.missing_values.values())
    metric_cols = st.columns(5)
    metric_cols[0].metric("Loaded rows", f"{preview_profile.row_count:,}")
    metric_cols[1].metric("Columns", preview_profile.column_count)
    metric_cols[2].metric("Numeric", len(preview_profile.numeric_columns))
    metric_cols[3].metric("Categorical", len(preview_profile.categorical_columns))
    metric_cols[4].metric("Missing values", missing_total)

    preview_col, schema_col = st.columns([1.75, 1])
    with preview_col:
        with st.container(border=True):
            st.markdown("### Dataset Preview")
            st.dataframe(df.head(8), width="stretch", height=300)
    with schema_col:
        with st.container(border=True):
            st.markdown("### Detected Schema")
            schema_rows = []
            for column in preview_profile.columns:
                if column in preview_profile.numeric_columns:
                    role = "numeric"
                elif column in preview_profile.date_columns:
                    role = "date"
                elif column in preview_profile.categorical_columns:
                    role = "categorical/text"
                else:
                    role = "other"
                schema_rows.append(
                    {
                        "column": column,
                        "role": role,
                        "missing": preview_profile.missing_values.get(column, 0),
                    }
                )
            st.dataframe(pd.DataFrame(schema_rows), width="stretch", height=300)

    run_disabled = bool(loaded_csv.error or preview_profile.row_count == 0)
    run_clicked = st.button("Run Agent Analysis", type="primary", disabled=run_disabled, use_container_width=True)
    if run_clicked:
        progress = st.progress(0, text="Initializing agent workflow...")
        with st.status("Agent workflow running", expanded=True) as status:
            status.write("Perceiving dataset state...")
            progress.progress(20, text="Perceiving dataset state")
            status.write("Planning safe tool sequence...")
            progress.progress(42, text="Planning safe tool sequence")
            status.write("Executing safe pandas and matplotlib tools...")
            progress.progress(68, text="Executing safe tools")
            agent = DataAnalysisAgent(OpenAICompatibleClient(config))
            st.session_state.last_run = agent.run(
                df,
                goal,
                st.session_state.tool_scores,
                prior_context=st.session_state.conversation_memory,
            )
            status.write("Synthesizing grounded answer and updating memory...")
            progress.progress(92, text="Synthesizing final answer")
            st.session_state.last_run_dataset_id = loaded_csv.source_id
            st.session_state.last_run_request_id = current_request_id
            st.session_state.conversation_memory = (
                st.session_state.conversation_memory
                + [
                    {
                        "goal": goal,
                        "final_answer": st.session_state.last_run.final_answer,
                        "tools": st.session_state.last_run.tools_used,
                    }
                ]
            )[-3:]
            progress.progress(100, text="Agent analysis complete")
            status.update(label="Agent analysis complete", state="complete", expanded=False)

run = st.session_state.last_run
if (
    run
    and st.session_state.last_run_dataset_id == st.session_state.active_dataset_id
    and st.session_state.last_run_request_id == st.session_state.active_request_id
):
    clarification = getattr(
        run,
        "clarification",
        {"ambiguous": False, "suggestions": [], "planning_goal": run.goal},
    )
    guardrail = getattr(run, "guardrail", {"blocked": False, "reason": "", "matched_terms": []})

    section_label("Agent Workspace")
    workspace_cols = st.columns(4)
    workspace_cols[0].metric("Plan source", run.plan.source)
    workspace_cols[1].metric("Tools used", len(run.tool_results))
    workspace_cols[2].metric("Trace events", len(run.trace))
    workspace_cols[3].metric("Guardrail", "Blocked" if guardrail["blocked"] else "Clear")

    answer_col, trace_col = st.columns([1.2, 1])
    with answer_col:
        with st.container(border=True):
            st.markdown("### Final Answer")
            st.write(run.final_answer)
    with trace_col:
        with st.container(border=True):
            st.markdown("### Agent Trace")
            trace_tabs = st.tabs(["Perceive", "Plan", "Act", "Observe"])
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
                if run.plan.error:
                    st.warning(f"LLM planner fallback reason: {run.plan.error}")
                st.dataframe(pd.DataFrame([step.to_dict() for step in run.plan.steps]), width="stretch")
                if run.plan.raw_response:
                    with st.expander("Raw LLM planner response"):
                        st.code(run.plan.raw_response)
            with trace_tabs[2]:
                for result in run.tool_results:
                    st.markdown(f"**{result.title}**")
                    st.write(result.observation)
            with trace_tabs[3]:
                st.json(run.trace)

    section_label("Tool Outputs")
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
        feedback_col, memory_col = st.columns([1, 1])
        with feedback_col:
            with st.container(border=True):
                st.markdown("### Feedback")
                useful_col, not_useful_col = st.columns(2)
                with useful_col:
                    if st.button("Useful", use_container_width=True):
                        st.session_state.tool_scores = update_tool_scores(
                            st.session_state.tool_scores,
                            run.tools_used,
                            "useful",
                        )
                        st.rerun()
                with not_useful_col:
                    if st.button("Not useful", use_container_width=True):
                        st.session_state.tool_scores = update_tool_scores(
                            st.session_state.tool_scores,
                            run.tools_used,
                            "not_useful",
                        )
                        st.rerun()
        with memory_col:
            with st.container(border=True):
                st.markdown("### Conversation Memory")
                st.json(st.session_state.conversation_memory)

section_label("Evaluation Dashboard")
with st.container(border=True):
    st.markdown("### Deterministic Evaluation Suite")
    st.caption("Runs edge-case checks with the LLM disabled, so the score reflects policy, guardrails, tools, and fallback behavior.")
    if st.button("Run Evaluation Suite", use_container_width=True):
        st.session_state.evaluation_results = run_evaluation(use_llm=False)

    evaluation_results = st.session_state.evaluation_results
    if evaluation_results:
        passed_count = sum(result.passed for result in evaluation_results)
        total_count = len(evaluation_results)
        pass_rate = passed_count / total_count if total_count else 0
        total_tools = sum(len(result.tools) for result in evaluation_results)
        total_charts = sum(result.chart_count for result in evaluation_results)
        average_latency = (
            sum(result.duration_ms for result in evaluation_results) / total_count
            if total_count
            else 0
        )
        blocked_count = sum(result.blocked for result in evaluation_results)
        clarification_count = sum(result.needs_clarification for result in evaluation_results)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Passed cases", f"{passed_count}/{total_count}")
        metric_cols[1].metric("Pass rate", f"{pass_rate:.0%}")
        metric_cols[2].metric("Avg latency", f"{average_latency:.0f} ms")
        metric_cols[3].metric("Tools / charts", f"{total_tools} / {total_charts}")

        coverage_cols = st.columns(3)
        coverage_cols[0].metric("Failed cases", str(total_count - passed_count))
        coverage_cols[1].metric("Guardrail cases", str(blocked_count))
        coverage_cols[2].metric("Clarification cases", str(clarification_count))

        if passed_count == total_count:
            st.success("All evaluation scenarios passed.")
        else:
            st.warning("Some scenarios need attention. Check the issues column.")

        st.dataframe(pd.DataFrame([result.to_dict() for result in evaluation_results]), width="stretch")
