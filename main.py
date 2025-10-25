from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from chains import (
    continuation_chain,
    first_responder_model,
    first_responder_parser,
    reflection_chain,
    revision_model,
    revision_parser,
)
from tool_executor import execute_tools


class AgentState(TypedDict):
    """Mutable state threaded through the reflexion workflow."""

    objective: str
    messages: List[BaseMessage]
    iteration: int
    max_iterations: int
    structured_result: Dict[str, Any]
    reflections: List[Dict[str, Any]]
    tool_logs: List[str]
    tool_results: List[Any]
    search_queries: List[str]
    decision: Optional[Dict[str, Any]]
    last_reflection_summary: Optional[Dict[str, Any]]


def draft(state: AgentState) -> AgentState:
    """Generate or revise the answer depending on the current iteration."""

    is_initial_iteration = state["iteration"] == 0
    model = first_responder_model if is_initial_iteration else revision_model
    parser = first_responder_parser if is_initial_iteration else revision_parser

    ai_message = model.invoke({"messages": state["messages"]})
    parsed = parser.invoke(ai_message)

    if isinstance(parsed, list):
        structured = parsed[0]
    else:
        structured = parsed

    structured_dict = structured.model_dump()

    messages = [*state["messages"], ai_message]

    return {
        **state,
        "messages": messages,
        "structured_result": structured_dict,
        "search_queries": structured_dict.get("search_queries", []),
        "tool_logs": [],
        "tool_results": [],
    }


def execute_research(state: AgentState) -> AgentState:
    """Run Tavily searches based on the model's suggested queries."""

    previous_messages = state["messages"]
    previous_count = len(previous_messages)
    tool_output = execute_tools.invoke(previous_messages)

    if isinstance(tool_output, list):
        if previous_count <= len(tool_output) and tool_output[:previous_count] == list(previous_messages):
            messages = tool_output
            new_messages = tool_output[previous_count:]
        else:
            new_messages = tool_output
            messages = [*previous_messages, *tool_output]
    else:
        new_messages = [tool_output]
        messages = [*previous_messages, tool_output]

    observations: List[str] = []
    results: List[Any] = []

    for message in new_messages:
        content = message.content
        results.append(content)
        if isinstance(content, str):
            observations.append(content)
        else:
            observations.append(str(content))

    if not observations:
        observations.append("Search tools returned no content.")

    return {
        **state,
        "messages": messages,
        "tool_logs": observations,
        "tool_results": results,
    }


def reflect(state: AgentState) -> AgentState:
    """Summarize the most recent critique and tool findings into actionable reflection."""

    structured = state["structured_result"]
    reflection_fields = structured["reflection"]
    tool_observations = "\n".join(state.get("tool_logs", [])) or "No tool observations available."

    summary = reflection_chain.invoke(
        {
            "objective": state["objective"],
            "iteration": state["iteration"] + 1,
            "answer": structured["answer"],
            "missing": reflection_fields["missing"],
            "superfluous": reflection_fields["superfluous"],
            "tool_observations": tool_observations,
        }
    )

    summary_dict = summary.model_dump()

    return {
        **state,
        "reflections": [*state["reflections"], summary_dict],
        "last_reflection_summary": summary_dict,
    }


def decide(state: AgentState) -> AgentState:
    """Determine whether to continue iterating and craft guidance for the next loop."""

    reflection = state["last_reflection_summary"] or {
        "highlights": ["No highlights captured."],
        "risks": ["No risks captured."],
        "next_focus": "Maintain current direction.",
        "confidence": 0.5,
    }

    decision = continuation_chain.invoke(
        {
            "objective": state["objective"],
            "iteration": state["iteration"] + 1,
            "max_iterations": state["max_iterations"],
            "highlights": ", ".join(reflection.get("highlights", [])),
            "risks": ", ".join(reflection.get("risks", [])),
            "next_focus": reflection.get("next_focus", "Maintain current direction."),
            "confidence": reflection.get("confidence", 0.5),
        }
    )

    decision_dict = decision.model_dump()

    reached_limit = (state["iteration"] + 1) >= state["max_iterations"]
    if reached_limit:
        decision_dict["should_continue"] = False
        decision_dict["reason"] = (
            f"{decision_dict['reason']} (Reached max iterations)"
            if decision_dict["reason"]
            else "Reached max iterations."
        )

    messages = state["messages"]
    if decision_dict["should_continue"]:
        guidance = (
            "Please revise the previous answer focusing on: "
            f"{decision_dict['next_focus']}. Rationale: {decision_dict['reason']}"
        )
        messages = [
            *messages,
            HumanMessage(content=guidance),
        ]

    return {
        **state,
        "decision": decision_dict,
        "messages": messages,
        "iteration": state["iteration"] + 1,
    }


def route_next(state: AgentState) -> str:
    return "draft" if state["decision"] and state["decision"]["should_continue"] else END


def build_graph() -> StateGraph[AgentState]:
    workflow: StateGraph[AgentState] = StateGraph(AgentState)
    workflow.add_node("draft", draft)
    workflow.add_node("research", execute_research)
    workflow.add_node("reflect", reflect)
    workflow.add_node("decide", decide)

    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "research")
    workflow.add_edge("research", "reflect")
    workflow.add_edge("reflect", "decide")

    workflow.add_conditional_edges(
        "decide",
        route_next,
        {
            "draft": "draft",
            END: END,
        },
    )

    return workflow


def main() -> None:
    load_dotenv()

    graph = build_graph()
    app = graph.compile()

    objective = (
        "Summarize a complex topic of your choice, highlighting current trends, notable players, "
        "and open questions that deserve further investigation."
    )

    initial_state: AgentState = {
        "objective": objective,
        "messages": [HumanMessage(content=objective)],
        "iteration": 0,
        "max_iterations": 3,
        "structured_result": {},
        "reflections": [],
        "tool_logs": [],
    "tool_results": [],
        "search_queries": [],
        "decision": None,
        "last_reflection_summary": None,
    }

    final_state = app.invoke(initial_state)

    final_answer = final_state.get("structured_result", {}).get(
        "answer", "No answer produced."
    )

    print("Final assistant answer:\n")
    print(final_answer)

    if final_state.get("tool_logs"):
        print("\nLatest tool observations:")
        for observation in final_state["tool_logs"]:
            print(f"- {observation}")

    if final_state.get("reflections"):
        print("\nReflection log:")
        for idx, reflection in enumerate(final_state["reflections"], start=1):
            highlights = "; ".join(reflection.get("highlights", []))
            risks = "; ".join(reflection.get("risks", []))
            next_focus = reflection.get("next_focus", "N/A")
            confidence = reflection.get("confidence", "N/A")
            print(
                f"\nIteration {idx}:\n"
                f"Highlights: {highlights}\n"
                f"Risks: {risks}\n"
                f"Next focus: {next_focus}\n"
                f"Confidence: {confidence}"
            )

    print("\nContinuation decision:")
    print(final_state.get("decision"))

    print("\nMermaid workflow diagram:\n")
    print(f"```mermaid\n{app.get_graph().draw_mermaid()}\n```")


if __name__ == "__main__":
    main()