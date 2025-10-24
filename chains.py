"""Shared prompt and chain utilities for the LangGraph application."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a reflective strategist distilling lessons learned. "
                "Diagnose what worked, what stalled, and spotlight surprising signals."
            ),
        ),
        MessagesPlaceholder("conversation"),
        (
            "human",
            (
                "In three vivid bullet points, capture the breakthroughs, blind spots, "
                "and unanswered curiosities that emerged while pursuing {objective}. "
                "Close with a mantra titled 'North Star' that keeps the team aligned."
            ),
        ),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a visionary builder turning reflection into momentum. Combine "
                "insight with bold but practical next moves tailored to {audience}."
            ),
        ),
        MessagesPlaceholder("reflections"),
        MessagesPlaceholder("conversation"),
        (
            "human",
            (
                "Draft the next assistant message that advances {objective}. "
                "Blend a {tone} tone with concrete, testable steps, and invite feedback "
                "that keeps the loop learning."
            ),
        ),
    ]
)

# Ensure environment variables are available before the model initializes.
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
# Parser ensures downstream graph logic receives plain text.
_parser = StrOutputParser()

reflection_chain = reflection_prompt | llm | _parser
generation_chain = generation_prompt | llm | _parser

__all__ = [
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "ChatOpenAI",
    "reflection_prompt",
    "generation_prompt",
    "llm",
    "reflection_chain",
    "generation_chain",
]
