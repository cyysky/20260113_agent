# AI Agents

A collection of CLI agents that use LiteLLM with Function Calling to perform various tasks.

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

## Requirements

- Python 3.8+
- litellm >= 1.0.0
- httpx >= 0.25.0 (for web agent)# 20260113_agent
