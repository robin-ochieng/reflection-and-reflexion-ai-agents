from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from chains import generation_chain, reflection_chain


class AgentState(TypedDict):
    conversation: List[BaseMessage]
    reflections: List[BaseMessage]
    objective: str
    audience: str
    tone: str
    turn: int
    max_turns: int


def generate(state: AgentState) -> AgentState:
    response = generation_chain.invoke(
        {
            "objective": state["objective"],
            "audience": state["audience"],
            "tone": state["tone"],
            "conversation": state["conversation"],
            "reflections": state["reflections"],
        }
    )

    return {
        **state,
        "conversation": [*state["conversation"], AIMessage(content=response)],
        "turn": state["turn"] + 1,
    }


def reflect(state: AgentState) -> AgentState:
    reflection = reflection_chain.invoke(
        {
            "objective": state["objective"],
            "conversation": state["conversation"],
        }
    )

    return {
        **state,
        "reflections": [*state["reflections"], AIMessage(content=reflection)],
    }


def route_should_continue(state: AgentState) -> str:
    if state["turn"] >= state["max_turns"]:
        return "end"
    if not state["reflections"]:
        return "reflect"
    return "end"


def build_graph() -> StateGraph[AgentState]:
    # Encodes the loop: start -> generate -> should_continue -> reflect -> generate/end.
    workflow: StateGraph[AgentState] = StateGraph(AgentState)
    workflow.add_node("generate", generate)
    workflow.add_node("should_continue", lambda state: state)
    workflow.add_node("reflect", reflect)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "should_continue")
    workflow.add_conditional_edges(
        "should_continue",
        route_should_continue,
        {
            "reflect": "reflect",
            "end": END,
        },
    )
    workflow.add_edge("reflect", "generate")

    return workflow


def main() -> None:
    load_dotenv()

    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "conversation": [
            HumanMessage(
                content="We need a launch plan for the reflection agent project that inspires the team."
            )
        ],
        "reflections": [],
        "objective": "Deliver the next assistant update about the reflection agent launch plan.",
        "audience": "core product team",
        "tone": "motivational",
        "turn": 0,
        "max_turns": 3,
    }

    final_state = app.invoke(initial_state)

    final_message = final_state["conversation"][-1].content
    print("Final assistant message:\n")
    print(final_message)

    if final_state["reflections"]:
        print("\nCaptured reflections:\n")
        for idx, reflection in enumerate(final_state["reflections"], start=1):
            print(f"Reflection {idx}: {reflection.content}\n")

    print("Mermaid workflow diagram:\n")
    print(f"```mermaid\n{app.get_graph().draw_mermaid()}\n```")


if __name__ == "__main__":
    main()
