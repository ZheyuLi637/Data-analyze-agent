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
DEFAULT_MAX_ANALYSIS_ROWS = 5000


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
        :root {
            --bg: #F7F9FB;
            --panel: #FFFFFF;
            --panel-solid: #FFFFFF;
            --line: #E1E4E8;
            --line-soft: #EEF1F4;
            --title: #111111;
            --text: #1F2328;
            --muted: #6E7781;
            --accent: #0969DA;
            --accent-soft: #DDF4FF;
            --success: #1A7F37;
            --warning: #9A6700;
            --shadow: 0 14px 36px rgba(27, 31, 36, 0.08);
            --shadow-soft: 0 8px 22px rgba(27, 31, 36, 0.06);
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
        }
        .block-container {
            position: relative;
            padding-top: 1.35rem;
            padding-bottom: 3.5rem;
            max-width: 1200px;
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
            box-shadow: 8px 0 30px rgba(27, 31, 36, 0.04);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--title);
            letter-spacing: 0;
        }
        h1, h2, h3 {
            color: var(--title);
            letter-spacing: 0;
        }
        .hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.15rem;
            padding: 2.15rem 2.15rem 1.9rem 2.15rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background:
                linear-gradient(135deg, #FFFFFF 0%, #F6F8FA 58%, #EDF5FF 100%);
            box-shadow: var(--shadow);
        }
        .hero::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 42%;
            height: 100%;
            pointer-events: none;
            background: radial-gradient(circle at 70% 26%, rgba(9, 105, 218, 0.12), transparent 18rem);
        }
        .hero h1 {
            color: var(--title);
            position: relative;
            font-size: 2.45rem;
            line-height: 1.08;
            margin: 0 0 0.55rem 0;
            letter-spacing: 0;
            font-weight: 760;
        }
        .hero p {
            position: relative;
            color: var(--muted);
            margin: 0;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 760px;
        }
        .eyebrow {
            position: relative;
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 720;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .section-label {
            color: var(--muted);
            font-size: 0.73rem;
            letter-spacing: 0;
            text-transform: uppercase;
            font-weight: 720;
            margin: 2rem 0 0.6rem 0;
        }
        .soft-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.55rem;
        }
        .window-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.2rem 0 0.62rem 0;
            border-bottom: 1px solid var(--line-soft);
            margin-bottom: 0.8rem;
        }
        .window-title {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 720;
        }
        .window-status {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.18rem 0.5rem;
            border: 1px solid #B6E3C6;
            border-radius: 999px;
            background: #DAFBE1;
            color: var(--success);
            font-size: 0.76rem;
            font-weight: 680;
        }
        .player-strip {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 620;
            padding-top: 0.65rem;
        }
        .player-dot {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 999px;
            background: var(--success);
            box-shadow: 0 0 0 3px #DAFBE1;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            box-shadow: var(--shadow-soft);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 650;
        }
        div[data-testid="stMetricValue"] {
            color: var(--title);
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid var(--line);
            background: #FFFFFF;
            transition: box-shadow 160ms ease, transform 160ms ease, border-color 160ms ease;
        }
        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #FFFFFF;
            box-shadow: 0 10px 22px rgba(9, 105, 218, 0.18);
        }
        .stButton > button:hover,
        .stButton > button:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.14);
            transform: translateY(-1px);
        }
        .stTextArea textarea {
            background: #FFFFFF;
            border: 0;
            box-shadow: none;
            color: var(--text);
            font-size: 1rem;
            line-height: 1.55;
        }
        .stTextArea textarea:focus {
            border: 0;
            box-shadow: none;
        }
        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            background: var(--panel-solid);
        }
        div[data-testid="stStatusWidget"] {
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            background: #FFFFFF !important;
            box-shadow: var(--shadow-soft);
        }
        .stProgress > div > div > div > div {
            background: var(--accent);
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
            <div class="eyebrow">Professional AI analysis prototype</div>
            <h1>Data Insight Agent</h1>
            <p>Upload a CSV, describe the decision you want to support, and inspect every step of the agent's perception, plan, tool actions, observations, and feedback loop.</p>
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

task_col, context_col = st.columns([2.1, 1])
with task_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="window-bar">
                <div class="window-title">Agent request</div>
                <div class="window-status">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Ask the agent")
        goal = st.text_area(
            "Agent goal",
            height=124,
            key="goal_text",
            label_visibility="collapsed",
            placeholder="Example: Compare profit by customer segment and explain the strongest group.",
        )
        st.markdown(
            '<div class="soft-note">Describe the decision you want the agent to support. Follow-up prompts can reuse recent memory.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="player-strip"><span class="player-dot"></span><span>Ready for perceive → plan → act → observe</span></div>',
            unsafe_allow_html=True,
        )

loaded_csv = read_input_data(uploaded, header_mode, DEFAULT_MAX_ANALYSIS_ROWS)
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
        st.markdown("### Runtime context")
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

    run_disabled = bool(loaded_csv.error or preview_profile.row_count == 0)
    section_label("Agent Control")
    with st.container(border=True):
        control_col, status_col = st.columns([1.2, 1])
        with control_col:
            run_clicked = st.button("Run Agent Analysis", type="primary", disabled=run_disabled, use_container_width=True)
        with status_col:
            if run_disabled:
                st.caption("Upload a CSV with at least one data row before running.")
            else:
                st.caption(f"Ready to analyze {preview_profile.row_count:,} rows and {preview_profile.column_count} columns.")
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
