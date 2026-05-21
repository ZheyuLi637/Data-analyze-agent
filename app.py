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
        :root {
            --bg: #090a1f;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-solid: #FFFFFF;
            --line: rgba(136, 231, 255, 0.42);
            --line-soft: rgba(255, 255, 255, 0.56);
            --title: #101224;
            --text: #22243a;
            --muted: #5c6078;
            --accent: #00d8ff;
            --pink: #ff4fd8;
            --purple: #8f5cff;
            --teal: #00e6c3;
            --chrome: linear-gradient(135deg, #ffffff 0%, #dbe7ff 20%, #ffdbfb 44%, #b9fff5 70%, #ffffff 100%);
        }
        .stApp {
            background:
                linear-gradient(rgba(255, 255, 255, 0.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.055) 1px, transparent 1px),
                radial-gradient(circle at 12% 8%, rgba(255, 79, 216, 0.42), transparent 20rem),
                radial-gradient(circle at 84% 5%, rgba(0, 216, 255, 0.4), transparent 22rem),
                radial-gradient(circle at 55% 35%, rgba(143, 92, 255, 0.28), transparent 24rem),
                #090a1f;
            background-size: 34px 34px, 34px 34px, auto, auto, auto, auto;
            color: var(--text);
            font-family: "Arial Narrow", "Roboto Condensed", -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
                repeating-linear-gradient(180deg, rgba(255, 255, 255, 0.035) 0px, rgba(255, 255, 255, 0.035) 1px, transparent 2px, transparent 6px);
            mix-blend-mode: screen;
        }
        .stApp::after {
            content: "✦";
            position: fixed;
            top: 5.5rem;
            right: 9vw;
            color: rgba(255, 255, 255, 0.7);
            font-size: 1.7rem;
            pointer-events: none;
            text-shadow: 0 0 18px rgba(0, 216, 255, 0.95);
        }
        .block-container {
            position: relative;
            z-index: 1;
            padding-top: 1.15rem;
            padding-bottom: 3.5rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(235, 249, 255, 0.9)),
                var(--panel-solid);
            border-right: 1px solid rgba(255, 255, 255, 0.6);
            box-shadow: 12px 0 36px rgba(0, 0, 0, 0.18);
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
        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0 0 1rem 0;
            padding: 0.68rem 0.78rem;
            border: 1px solid rgba(255, 255, 255, 0.78);
            border-radius: 18px;
            background: var(--chrome);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.95),
                0 0 0 1px rgba(0, 216, 255, 0.18),
                0 18px 42px rgba(7, 10, 45, 0.24);
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: #111225;
            font-weight: 800;
            font-size: 0.98rem;
        }
        .brand-dot {
            width: 0.82rem;
            height: 0.82rem;
            border-radius: 999px;
            background: radial-gradient(circle at 30% 30%, #ffffff, var(--teal) 40%, var(--purple) 100%);
            box-shadow: 0 0 16px rgba(0, 216, 255, 0.95);
        }
        .nav-links {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            flex-wrap: wrap;
        }
        .nav-chip {
            padding: 0.28rem 0.55rem;
            border: 1px solid rgba(255, 255, 255, 0.66);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.44);
            color: #1f2340;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
            padding: 2rem 2rem 1.75rem 2rem;
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(235, 244, 255, 0.84) 46%, rgba(255, 224, 250, 0.86)),
                var(--panel-solid);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.95),
                0 24px 70px rgba(4, 7, 35, 0.24),
                0 0 36px rgba(0, 216, 255, 0.12);
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(90deg, rgba(0, 216, 255, 0.16), transparent 28%, rgba(255, 79, 216, 0.16)),
                repeating-linear-gradient(90deg, transparent 0, transparent 28px, rgba(143, 92, 255, 0.08) 29px);
        }
        .hero h1 {
            color: var(--title);
            position: relative;
            font-size: 2.8rem;
            line-height: 1.08;
            margin: 0 0 0.45rem 0;
            letter-spacing: 0;
            font-weight: 900;
            text-shadow: 0 0 18px rgba(255, 79, 216, 0.24);
        }
        .hero p {
            position: relative;
            color: var(--muted);
            margin: 0;
            font-size: 1.02rem;
            max-width: 760px;
        }
        .eyebrow {
            position: relative;
            color: #3a2f86;
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0.25rem 0;
        }
        .feature-card {
            border: 1px solid rgba(255, 255, 255, 0.64);
            border-radius: 18px;
            padding: 0.9rem 0.95rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.82), rgba(225, 255, 251, 0.58)),
                rgba(255, 255, 255, 0.82);
            box-shadow: 0 16px 42px rgba(10, 11, 45, 0.18);
        }
        .feature-card strong {
            display: block;
            color: #15172a;
            font-size: 0.95rem;
            margin-bottom: 0.15rem;
        }
        .feature-card span {
            color: #5b5f77;
            font-size: 0.84rem;
        }
        .section-label {
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.73rem;
            letter-spacing: 0;
            text-transform: uppercase;
            font-weight: 850;
            margin: 2rem 0 0.6rem 0;
            text-shadow: 0 0 14px rgba(0, 216, 255, 0.6);
        }
        .soft-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.55rem;
        }
        .input-shell {
            background:
                linear-gradient(var(--panel-solid), var(--panel-solid)) padding-box,
                linear-gradient(135deg, var(--pink), var(--accent), var(--teal)) border-box;
            border: 1px solid transparent;
            border-radius: 22px;
            padding: 0.75rem 1.05rem 1rem 1.05rem;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.9),
                0 18px 46px rgba(0, 0, 0, 0.24),
                0 0 26px rgba(0, 216, 255, 0.2);
        }
        .window-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.2rem 0 0.62rem 0;
            border-bottom: 1px solid rgba(225, 228, 232, 0.8);
            margin-bottom: 0.8rem;
        }
        .window-title {
            color: #25283d;
            font-size: 0.86rem;
            font-weight: 850;
        }
        .window-controls {
            display: flex;
            gap: 0.32rem;
        }
        .window-controls span {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
            display: inline-block;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
        }
        .window-controls span:nth-child(1) { background: var(--pink); }
        .window-controls span:nth-child(2) { background: var(--accent); }
        .window-controls span:nth-child(3) { background: var(--teal); }
        .player-strip {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: #383a52;
            font-size: 0.8rem;
            font-weight: 750;
            padding-top: 0.65rem;
        }
        .player-dot {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 999px;
            background: var(--teal);
            box-shadow: 0 0 12px var(--teal);
        }
        .context-card {
            background:
                linear-gradient(135deg, rgba(255,255,255,0.92), rgba(238, 246, 255, 0.84));
            border: 1px solid rgba(255, 255, 255, 0.74);
            border-radius: 22px;
            padding: 1rem;
            min-height: 100%;
            box-shadow: 0 18px 46px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247, 250, 255, 0.94));
            border: 1px solid rgba(255, 255, 255, 0.72) !important;
            border-radius: 20px !important;
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.9),
                0 18px 42px rgba(5, 8, 38, 0.18);
        }
        div[data-testid="stMetric"] {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.94), rgba(235, 249, 255, 0.86));
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 12px 32px rgba(5, 8, 38, 0.16);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 650;
        }
        div[data-testid="stMetricValue"] {
            color: var(--title);
        }
        .stButton > button {
            border-radius: 16px;
            font-weight: 700;
            border: 1px solid rgba(255, 255, 255, 0.72);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.94), rgba(230, 244, 255, 0.9));
            transition: box-shadow 160ms ease, transform 160ms ease, border-color 160ms ease;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #ff4fd8 0%, #8f5cff 48%, #00d8ff 100%);
            border-color: rgba(255,255,255,0.86);
            color: #FFFFFF;
            box-shadow:
                0 0 0 1px rgba(255,255,255,0.36),
                0 0 20px rgba(0, 216, 255, 0.46),
                0 16px 34px rgba(76, 42, 199, 0.34);
        }
        .stButton > button:hover,
        .stButton > button:focus {
            border-color: rgba(0, 216, 255, 0.9);
            box-shadow:
                0 0 0 2px rgba(255, 79, 216, 0.16),
                0 0 22px rgba(0, 216, 255, 0.46);
            transform: translateY(-1px);
        }
        .stTextArea textarea {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249, 252, 255, 0.98));
            border: 0;
            box-shadow: none;
            color: var(--text);
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .stTextArea textarea:focus {
            border: 0;
            box-shadow: none;
        }
        .stDataFrame {
            border: 1px solid rgba(225, 228, 232, 0.9);
            border-radius: 16px;
            overflow: hidden;
            background: var(--panel-solid);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
        }
        div[data-testid="stStatusWidget"] {
            border: 1px solid rgba(0, 216, 255, 0.36) !important;
            border-radius: 18px !important;
            background: rgba(255,255,255,0.92) !important;
            box-shadow: 0 0 24px rgba(0, 216, 255, 0.16);
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--pink), var(--purple), var(--accent));
        }
        @media (max-width: 900px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }
            .top-nav {
                align-items: flex-start;
                flex-direction: column;
            }
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
        <div class="top-nav">
            <div class="brand-lockup"><span class="brand-dot"></span><span>COMPSCI 767 Agent OS</span></div>
            <div class="nav-links">
                <span class="nav-chip">CSV Upload</span>
                <span class="nav-chip">Safe Tools</span>
                <span class="nav-chip">Trace</span>
                <span class="nav-chip">Evaluation</span>
            </div>
        </div>
        <div class="hero">
            <div class="eyebrow">Y2K AI data workspace</div>
            <h1>Data Insight Agent</h1>
            <p>Upload a CSV, describe the decision you want to support, and inspect every step of the agent's perception, plan, tool actions, observations, and feedback loop.</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card"><strong>✦ Perception Matrix</strong><span>Detect schema, missing values, text, dates, and column intent before planning.</span></div>
            <div class="feature-card"><strong>◈ Tool Runner</strong><span>Validated pandas and matplotlib actions execute without arbitrary code access.</span></div>
            <div class="feature-card"><strong>▣ Evaluation Deck</strong><span>Edge-case checks track guardrails, clarification, latency, charts, and pass rate.</span></div>
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
    st.markdown('<div class="input-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="window-bar">
            <div class="window-title">AGENT_COMMAND.EXE</div>
            <div class="window-controls"><span></span><span></span><span></span></div>
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
    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown('<div class="context-card">', unsafe_allow_html=True)
    st.markdown("### Runtime context")
    st.metric("Memory turns", len(st.session_state.conversation_memory))
    st.metric("LLM mode", "Enabled" if config.ready else "Fallback")
    if loaded_csv.error:
        st.error("CSV not ready")
    else:
        assert preview_profile is not None
        st.caption(loaded_csv.source_name)
        st.caption(f"{preview_profile.row_count} loaded rows / {loaded_csv.original_row_count} source rows")
    st.markdown("</div>", unsafe_allow_html=True)

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
