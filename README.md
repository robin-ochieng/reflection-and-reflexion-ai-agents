# reflection-agent

Reflection-Agent is a LangGraph-powered prototype that experiments with reflexion agents—systems that pause to analyze their own behavior, extract lessons, and turn those insights into stronger follow-up actions. The current workflow pairs two LangChain `ChatPromptTemplate`s: one dedicated to reflective analysis and another for generating the next assistant response informed by the latest reflections.

## Key Features

- **Poetry-managed project** with pinned dependencies, development tooling, and reproducible virtual environment.
- **Reflection and generation chains** (`chains.py`) composed of creative prompts, a shared `ChatOpenAI` client, and a string parser ready for LangGraph integration.
- **LangGraph state machine** (`main.py`) that loops through `generate → should_continue → reflect → generate`, capturing insights and printing a Mermaid diagram of the active workflow.

## Getting Started

```powershell
poetry install
poetry run python main.py
```

Configure credentials in `.env` (e.g., `OPENAI_API_KEY`) before running. The script will print the final assistant message, any reflections collected during the run, and the Mermaid visualization of the graph.
