"""Streamlit dashboard for interacting with the reflexion agent."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

from main import AgentState, build_graph

load_dotenv()


def _initialize_app() -> None:
    if "graph" not in st.session_state:
        graph = build_graph()
        st.session_state.graph = graph.compile()
    if "last_state" not in st.session_state:
        st.session_state.last_state = None


def _render_messages(messages: List[BaseMessage]) -> None:
    with st.expander("Conversation Trace", expanded=False):
        for idx, message in enumerate(messages, start=1):
            role = getattr(message, "type", "assistant")
            content = getattr(message, "content", "")
            st.markdown(f"**{idx}. {role.title()}**\n\n{content}")


def _render_reflections(reflections: List[Dict]) -> None:
    if not reflections:
        st.info("No reflections captured yet.")
        return

    for idx, reflection in enumerate(reflections, start=1):
        with st.expander(f"Reflection {idx}", expanded=idx == len(reflections)):
            highlights = reflection.get("highlights", [])
            risks = reflection.get("risks", [])
            next_focus = reflection.get("next_focus", "N/A")
            confidence = reflection.get("confidence", "N/A")

            st.markdown("**Highlights**")
            for item in highlights:
                st.markdown(f"- {item}")

            st.markdown("**Risks**")
            for item in risks:
                st.markdown(f"- {item}")

            st.markdown("**Next Focus**")
            st.write(next_focus)

            st.markdown("**Confidence**")
            st.write(confidence)


def _invoke_agent(objective: str, max_iterations: int) -> Dict:
    initial_state: AgentState = {
        "objective": objective,
        "messages": [HumanMessage(content=objective)],
        "iteration": 0,
        "max_iterations": max_iterations,
        "structured_result": {},
        "reflections": [],
        "tool_logs": [],
        "search_queries": [],
        "decision": None,
        "last_reflection_summary": None,
    }

    return st.session_state.graph.invoke(initial_state)


def main() -> None:
    st.set_page_config(
        page_title="Reflexion Agent",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Reflexion Agent Workbench")
    st.write(
        "Experiment with a LangGraph reflexion workflow that drafts, reflects, and decides whether to iterate again."
    )

    _initialize_app()

    default_objective = (
        "Write about the AI-powered SOC / autonomous SOC problem domain and list startups that "
        "operate there with recent funding."
    )

    with st.form("agent_form"):
        objective = st.text_area("Objective", value=default_objective, height=160)
        max_iterations = st.slider("Max iterations", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Run agent", use_container_width=True)

    if submitted:
        with st.spinner("Running reflexion agent..."):
            final_state = _invoke_agent(objective, max_iterations)
        st.session_state.last_state = final_state

    if st.session_state.last_state is None:
        st.info("Configure an objective and click 'Run agent' to see results.")
        return

    state = st.session_state.last_state
    final_answer = state.get("structured_result", {}).get("answer", "No answer produced.")

    st.subheader("Final Answer")
    st.write(final_answer)

    st.subheader("Reflection Log")
    _render_reflections(state.get("reflections", []))

    st.subheader("Continuation Decision")
    decision = state.get("decision", {})
    if decision:
        st.json(decision)
    else:
        st.write("Decision data unavailable.")

    st.subheader("Mermaid Diagram")
    st.code(st.session_state.graph.get_graph().draw_mermaid(), language="mermaid")

    st.subheader("Debug")
    _render_messages(state.get("messages", []))

if __name__ == "__main__":
    main()