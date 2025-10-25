"""Tool execution utilities for the reflexion agent."""

from __future__ import annotations

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

_tavily = TavilySearch(max_results=5)


def run_queries(search_queries: List[str], **_: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute Tavily search queries in batch."""
    if not search_queries:
        return []
    payload = [{"query": query} for query in search_queries]
    return _tavily.batch(payload)


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

__all__ = ["execute_tools", "run_queries"]
