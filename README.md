# AI Agents

A collection of CLI agents that use LiteLLM with Function Calling to perform various tasks, with an Orchestrator to coordinate multiple agents.

## Agents

### File Agent

Manages files autonomously using AI function calling.

**Features:**
- AI-powered file management
- Function calling for file operations
- Secure path traversal protection
- Conversation history

**See:** [File Agent Documentation](file_agent.py)

### Web Agent

Browses the internet autonomously using AI function calling.

**Features:**
- AI-powered web search and page fetching
- Supports Google Custom Search and DuckDuckGo
- Extracts text content from URLs
- Multi-round tool calling support

**See:** [Web Agent Documentation](web_agent.py)

### Summary Agent

Creates well-formatted markdown summaries from raw research data.

**Features:**
- Converts raw search/fetch results into clean markdown
- Organizes content with proper structure
- Includes source citations

**See:** [Summary Agent Documentation](summary_agent.py)

### Orchestrator Agent

Coordinates multiple agents using LLM-planned workflows.

**Features:**
- **LLM-Planned Pipelines**: The LLM dynamically plans the best agent sequence for each request
- **Flexible Input Routing**: Each step knows what input to pass (user message, accumulated content, or tool results)
- **Extensible Design**: Easily add new agents - the LLM will consider them for planning
- **Fallback Planning**: Simple keyword-based planner if LLM planning fails
- Maintains conversation history across agents

**See:** [Orchestrator Documentation](orchestrator.py)

---

## Common Configuration

Set the following environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_BASEURL` | LiteLLM API base URL | (required) |
| `LITELLM_API_KEY` | API key for authentication | (required) |
| `LITELLM_MODEL` | Model name to use | `gpt-3.5-turbo-1106` |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LiteLLM settings
```

## Supported Models

Models that support function calling:

- `gpt-3.5-turbo-1106` and newer
- `gpt-4-turbo` and newer
- `openai/qwen3-vl-235b-a22b-instruct`
- `azure/chatgpt-functioncalling`
- `xai/grok-2-latest`
- Other function-calling supported models

## Orchestrator Usage

The Orchestrator uses LLM-powered workflow planning to handle complex multi-step tasks.

```bash
# Start the orchestrator
python orchestrator.py
```

**How it works:**
1. User submits a request
2. LLM analyzes the request and available agents
3. LLM creates a dynamic execution plan
4. Orchestrator executes each step in sequence

**Example commands:**
```
> search for 2025 ai news related to malaysia and save it into .md file
> find information about python tutorials and write to notes.md
> look up latest tech news and create a summary document
```

**Available commands in Orchestrator:**
- `/agents` - List all registered agents
- `/plan <query>` - Preview the planned agent pipeline for a query
- `/history` - Show conversation history
- `/clear` - Clear conversation history
- `/quit` - Exit

**Dynamic Planning:**
For any request, the LLM decides which agents to use and in what order. Example output:
```
[Request] search for 2025 ai news related to malaysia and save it into .md file
[Planning] LLM planned 3 steps
  1. web_agent: Research the topic and gather information from the web
  2. summary_agent: Create a well-formatted summary from the research
  3. file_agent: Write the summary to the specified file
```

## Adding New Agents

To add a new agent:

1. Create your agent module (e.g., `db_agent.py`)
2. Export these required components:
   - `SYSTEM_PROMPT` - Agent's system prompt
   - `TOOLS` - List of tool definitions
   - `AVAILABLE_FUNCTIONS` - Dict of function name -> callable
   - `chat_func(message, history, max_turns)` - Chat function

3. Register in orchestrator.py:
```python
def create_db_agent():
    import db_agent
    return Agent(
        name="db_agent",
        description="Database operations",
        system_prompt=db_agent.SYSTEM_PROMPT,
        tools=db_agent.TOOLS,
        available_functions=db_agent.AVAILABLE_FUNCTIONS,
        chat_func=db_agent.chat,
    )

orchestrator.register_agent(create_db_agent())
```

## Requirements

- Python 3.8+
- litellm >= 1.0.0
- httpx >= 0.25.0 (for web agent)
- python-dotenv >= 1.0.0
