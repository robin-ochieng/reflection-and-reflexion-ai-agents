# Reflexion Agent Architecture Overview

## 1. Project Goals
- Build a self-improving "reflexion" agent that iteratively drafts, critiques, and refines answers to complex objectives.
- Provide a reproducible Python environment (Poetry-managed) and a Streamlit UI so stakeholders can inspect the agent's reasoning loop.

## 2. High-Level Flow
```mermaid
graph TD;
    user(Objective Input) -->|via CLI / Streamlit| graph
    subgraph LangGraph Workflow
        draft
        research
        reflect
        decide
    end
    graph --> draft
    draft --> research
    research --> reflect
    reflect --> decide
    decide -->|should_continue| draft
    decide -->|complete| outputs
    outputs --> final_answer(Structured Final Answer)
    outputs --> reflections(Reflection Log)
    outputs --> decision(Continuation Decision)
```

1. **Objective ingestion** – User supplies a task through CLI (`main.py`) or Streamlit (`streamlit_app.py`).
2. **Draft** – Agent creates/updates a structured answer (`AnswerQuestion` / `ReviseAnswer`).
3. **Research (stubbed)** – Emits human-readable observations from suggested search queries; prepared for future tool integrations.
4. **Reflect** – Summarizes strengths, risks, and next focus via `ReflectionSummary` schema.
5. **Decide** – Evaluates confidence and iteration count via `ContinuationDecision` schema to determine another loop or finish.
6. **Outputs** – Final answer, reflection history, and decision rendered in terminal and UI.

## 3. Key Modules

| Layer | Files | Responsibilities |
| --- | --- | --- |
| **Prompts & Chains** | `chains.py` | Configures reusable `ChatPromptTemplate` pipelines for initial drafting, revision, reflection, and continuation decisions. Uses `langchain-openai` and structured output parsing. |
| **Schemas** | `schemas.py` | Defines Pydantic models (`AnswerQuestion`, `ReviseAnswer`, `ReflectionSummary`, `ContinuationDecision`) to enforce consistent structure across iterations. |
| **Workflow Orchestration** | `main.py` | Declares `AgentState` and builds a `StateGraph` with nodes `draft → research → reflect → decide`. Handles state mutation, loop logic, CLI execution, and Mermaid export. |
| **User Interface** | `streamlit_app.py` | Provides an interactive dashboard to run the agent, display the final answer, reflection log, continuation decision, Mermaid diagram, and debug conversation trace. |
| **Environment & Tooling** | `pyproject.toml`, `poetry.lock`, `.env`, `.gitignore` | Manage dependencies, interpreter version, secrets, and Git hygiene. |

## 4. State Management
- `AgentState` tracks objective, conversation messages, iteration counters, structured outputs, reflections, tool logs, and continuation decisions.
- Each node returns an updated state dictionary, enabling LangGraph to propagate context through the workflow.
- Reflections and decisions persist across turns so subsequent drafts address prior critiques.

## 5. Iteration Mechanics
- `first_responder_chain` handles iteration 0; `revision_chain` handles subsequent iterations, both returning structured tool outputs.
- Reflection condenses critiques into highlights/risks and raises or lowers confidence.
- Continuation logic checks both the decision schema and max-iteration guard, ensuring termination even if the model wants to continue indefinitely.

## 6. Extensibility Points
- **Tool integration**: Replace the stubbed research step with real web search, document retrieval, or domain-specific APIs.
- **Custom prompts**: Adjust prompts for different domains (e.g., marketing copy, legal analysis) without changing graph structure.
- **Analytics**: Persist reflections and decisions for longitudinal studies or agent benchmarking.

## 7. Running the System
```powershell
# Install dependencies
poetry install

# Run CLI workflow
poetry run python main.py

# Launch Streamlit dashboard
poetry run streamlit run streamlit_app.py
```

Ensure `.env` contains valid API keys (`OPENAI_API_KEY`, etc.) before execution. The Streamlit app is reachable at `http://localhost:8501` by default.

## 8. Deliverables Summary
- Structured LangGraph reflexion loop with confidence-based stopping.
- Reusable LangChain chains and Pydantic schemas.
- Streamlit UI for experimentation and visualization.
- Documentation and tooling for reproducible development.
