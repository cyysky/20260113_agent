# File Agent

A CLI agent that autonomously manages files using LiteLLM with Function Calling.

## Features

- **AI-powered file management** - The AI can autonomously list, read, and write files
- **Function calling** - Uses LiteLLM's function calling to invoke file operations
- **Secure** - Restricted to the configured folder with path traversal protection
- **Conversation history** - Maintains context across messages

## Configuration

Set the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_BASEURL` | LiteLLM API base URL | (required) |
| `LITELLM_API_KEY` | API key for authentication | (required) |
| `LITELLM_MODEL` | Model name to use | `gpt-3.5-turbo-1106` |
| `AI_FOLDER_PATH` | Folder path for file operations | `./ai_files` |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LiteLLM settings
```

## Available Functions

The AI can autonomously call these functions:

| Function | Description |
|----------|-------------|
| `list_files` | List files in the folder or subfolder |
| `read_file` | Read the contents of a file |
| `write_file` | Write content to a file |

## Usage

### Interactive Mode

```bash
python file_agent.py
```

### Programmatic Usage

```python
from file_agent import chat, list_files, read_file, write_file

# Simple function calls
print(list_files())
print(read_file("notes.txt"))
print(write_file("hello.txt", "Hello World"))

# AI chat with function calling
response, history = chat("List all files and create a summary.txt")
print(response)
```

## Examples

```
> List all files in the folder
The folder contains:
  [DIR]  documents
  [FILE] notes.txt

> Read the contents of notes.txt
Contents of notes.txt:
  Hello, this is a note!

> Create a new file called hello.txt with "Hello World"
Successfully wrote to hello.txt

> Show me what files are in the documents subfolder
Files in C:\workspace\20260109_agent\ai_files\documents:
  [FILE] report.txt
```

## How It Works

1. User sends a natural language request
2. AI analyzes the request and determines which functions to call
3. Functions are executed and results returned to the AI
4. AI provides a helpful response based on the results

### Function Calling Flow

```
User Request -> AI Model -> Tool Calls -> Execute Functions -> Results -> AI Response
```

## Supported Models

Models that support function calling:

- `gpt-3.5-turbo-1106` and newer
- `gpt-4-turbo` and newer
- `azure/chatgpt-functioncalling`
- `xai/grok-2-latest`
- Other function-calling supported models

Check with:
```python
from litellm import supports_function_calling
print(supports_function_calling(model="gpt-3.5-turbo-1106"))  # True
```

## Security

- The agent can only access files within `AI_FOLDER_PATH`
- Path traversal attacks are blocked (`../../../etc/passwd`)
- All file operations are restricted to the configured directory

## Requirements

- Python 3.8+
- litellm >= 1.0.0