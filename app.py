"""
Streamlit application for the multi-agent system.

This app exposes three high-level tasks—summarisation of
medical text, writing and refining research articles, and
sanitising medical data. It also exposes a secondary tab
that visualises recent traces produced by the agents. A
toggle in the sidebar allows switching between OpenAI and
Ollama backends on the fly.
"""
import json
import glob
import os
from typing import Any

import streamlit as st

from agents import AgentManager
from utils.logger import logger


def display_traces(max_files: int = 10) -> None:
    """Render a summary of recent trace files in the UI."""
    trace_files = sorted(glob.glob(os.path.join("traces", "*.jsonl")), reverse=True)[:max_files]
    if not trace_files:
        st.info("No trace files found. Run some tasks first to generate traces.")
        return
    for path in trace_files:
        st.caption(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            st.error(f"Failed to read {path}: {e}")
            continue
        for line in lines:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**Task ID:** `{ex.get('task_id', '')}` — **Verdict:** {ex.get('verdict', '')}")
            cols[1].metric("Latency (ms)", ex.get("latency_ms", 0))
            cols[2].metric("Prompt tokens", ex.get("cost_tokens_prompt", 0))
            cols[3].metric("Output tokens", ex.get("cost_tokens_output", 0))
            with st.expander("Messages"):
                for m in ex.get("messages", []):
                    role = m.get("role", "")
                    sender = m.get("sender", "")
                    content = m.get("content", "")
                    st.write(f"[{role}] **{sender}**: {content}")


def summarize_section(agent_manager: AgentManager) -> None:
    st.header("Summarize Medical Text")
    text = st.text_area("Enter medical text to summarize:", height=200)
    if st.button("Summarize"):
        if text:
            main_agent = agent_manager.get_agent("summarize")
            validator_agent = agent_manager.get_agent("summarize_validator")
            with st.spinner("Summarizing..."):
                try:
                    summary = main_agent.execute(text)
                    st.subheader("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"SummarizeAgent Error: {e}")
                    return
            with st.spinner("Validating summary..."):
                try:
                    validation = validator_agent.execute(original_text=text, summary=summary)
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"SummarizeValidatorAgent Error: {e}")
        else:
            st.warning("Please enter some text to summarize.")


def write_and_refine_article_section(agent_manager: AgentManager) -> None:
    st.header("Write and Refine Research Article")
    topic = st.text_input("Enter the topic for the research article:")
    outline = st.text_area("Enter an outline (optional):", height=150)
    if st.button("Write and Refine Article"):
        if topic:
            writer_agent = agent_manager.get_agent("write_article")
            refiner_agent = agent_manager.get_agent("refiner")
            validator_agent = agent_manager.get_agent("validator")
            with st.spinner("Writing article..."):
                try:
                    draft = writer_agent.execute(topic, outline)
                    st.subheader("Draft Article:")
                    st.write(draft)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"WriteArticleAgent Error: {e}")
                    return
            with st.spinner("Refining article..."):
                try:
                    refined_article = refiner_agent.execute(draft)
                    st.subheader("Refined Article:")
                    st.write(refined_article)
                except Exception as e:
                    st.error(f"Refinement Error: {e}")
                    logger.error(f"RefinerAgent Error: {e}")
                    return
            with st.spinner("Validating article..."):
                try:
                    validation = validator_agent.execute(topic=topic, article=refined_article)
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"ValidatorAgent Error: {e}")
        else:
            st.warning("Please enter a topic for the research article.")


def sanitize_data_section(agent_manager: AgentManager) -> None:
    st.header("Sanitize Medical Data (PHI)")
    medical_data = st.text_area("Enter medical data to sanitize:", height=200)
    if st.button("Sanitize Data"):
        if medical_data:
            main_agent = agent_manager.get_agent("sanitize_data")
            validator_agent = agent_manager.get_agent("sanitize_data_validator")
            with st.spinner("Sanitizing data..."):
                try:
                    sanitized_data = main_agent.execute(medical_data)
                    st.subheader("Sanitized Data:")
                    st.write(sanitized_data)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"SanitizeDataAgent Error: {e}")
                    return
            with st.spinner("Validating sanitized data..."):
                try:
                    validation = validator_agent.execute(original_data=medical_data, sanitized_data=sanitized_data)
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"SanitizeDataValidatorAgent Error: {e}")
        else:
            st.warning("Please enter medical data to sanitize.")


def main() -> None:
    st.set_page_config(page_title="Multi-Agent AI System", layout="wide")
    st.title("Multi-Agent AI System with Collaboration and Validation")
    # Sidebar settings
    st.sidebar.title("Settings")
    use_local = st.sidebar.toggle(
        "Use Ollama (local)",
        value=(os.environ.get("USE_OLLAMA", "false").lower() == "true"),
    )
    # Update environment variable based on toggle
    os.environ["USE_OLLAMA"] = "true" if use_local else "false"

    st.sidebar.title("Select Task")
    task = st.sidebar.selectbox(
        "Choose a task:",
        [
            "Summarize Medical Text",
            "Write and Refine Research Article",
            "Sanitize Medical Data (PHI)",
        ],
    )
    # Instantiate agent manager once per request
    agent_manager = AgentManager(max_retries=2, verbose=True)
    # Tabs for run and traces
    tab_run, tab_traces = st.tabs(["Run", "Traces"])
    with tab_run:
        if task == "Summarize Medical Text":
            summarize_section(agent_manager)
        elif task == "Write and Refine Research Article":
            write_and_refine_article_section(agent_manager)
        elif task == "Sanitize Medical Data (PHI)":
            sanitize_data_section(agent_manager)
    with tab_traces:
        display_traces()


if __name__ == "__main__":
    main()