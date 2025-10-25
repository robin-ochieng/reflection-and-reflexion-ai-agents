import datetime

from dotenv import load_dotenv

from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import (
    AnswerQuestion,
    ContinuationDecision,
    ReflectionSummary,
    ReviseAnswer,
)

load_dotenv()

llm = ChatOpenAI(model="o4-mini")


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """You are an adaptable expert collaborator supporting a reflexion agent across any topic.
Current time: {time}

1. {first_instruction}
2. Reflect critically on the draft by identifying what is missing or superfluous for the stated objective.
3. Recommend concrete search queries that can improve the next iteration regardless of subject area.""",
    ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Always follow the requested output schema and make your reasoning explicit for any domain.",
        ),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder_model = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
first_responder_parser = PydanticToolsParser(tools=[AnswerQuestion])

revise_instructions = """Revise your previous answer using the new information.
- Lean on your earlier critique to add essential information and remove fluff.
- You MUST include numerical citations in square brackets inside the answer.
- Append a "References" section (outside the word limit) listing the sources you relied on."""

revision_model = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)
revision_parser = PydanticToolsParser(tools=[ReviseAnswer])


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the reflective conscience of the agent. Summarize insights that will guide the next iteration.",
        ),
        (
            "human",
            """Objective: {objective}
Iteration: {iteration}

Latest answer:
{answer}

Critique - missing:
{missing}

Critique - superfluous:
{superfluous}

Tool observations:
{tool_observations}

Return bullet lists of highlights and risks, a single-sentence next focus, and a confidence score between 0 and 1.""",
        ),
    ]
)

reflection_chain = reflection_prompt | llm.with_structured_output(ReflectionSummary)


continuation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether the agent should continue iterating. Follow the continuation schema exactly.",
        ),
        (
            "human",
            """Objective: {objective}
Iteration: {iteration} of {max_iterations}

Highlights considered:
{highlights}

Risks still present:
{risks}

Proposed next focus:
{next_focus}

Confidence the current answer is launch-ready: {confidence}

Should the agent continue refining its work?""",
        ),
    ]
)

continuation_chain = (
    continuation_prompt | llm.with_structured_output(ContinuationDecision)
)


__all__ = [
    "llm",
    "first_responder_model",
    "first_responder_parser",
    "revision_model",
    "revision_parser",
    "reflection_chain",
    "continuation_chain",
]