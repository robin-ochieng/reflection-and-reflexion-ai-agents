from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        default_factory=list,
        description="1-3 search queries for researching improvements to address the critique of your current answer.",
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        default_factory=list,
        description="Citations motivating your updated answer.",
    )


class ReflectionSummary(BaseModel):
    """Structured summary of the agent's self-reflection."""

    highlights: List[str] = Field(
        description="Key strengths or validated insights uncovered in the latest answer.",
        min_items=1,
    )
    risks: List[str] = Field(
        description="Issues, gaps, or uncertainties that still need attention.",
        min_items=1,
    )
    next_focus: str = Field(
        description="One concise directive the agent should prioritize in the next iteration."
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1 describing how close the draft is to final.",
        ge=0.0,
        le=1.0,
    )


class ContinuationDecision(BaseModel):
    """Decision on whether to continue iterating on the answer."""

    should_continue: bool = Field(
        description="True when the agent ought to run another improvement iteration."
    )
    reason: str = Field(
        description="Rationale that justifies the decision in natural language."
    )
    next_focus: str = Field(
        description="Focus statement the agent should follow if another iteration occurs."
    )