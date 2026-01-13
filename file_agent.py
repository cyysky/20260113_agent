"""
File Agent - List, read, and write files using LiteLLM with Function Calling

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use (e.g., gpt-4, gpt-3.5-turbo-1106)
- AI_FOLDER_PATH: The folder path the agent can access

The AI can autonomously call these functions:
- list_files: List files in the folder or subfolder
- read_file: Read the contents of a file
- write_file: Write content to a file
"""

import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from litellm import completion, supports_function_calling

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
BASE_URL = os.environ.get("LITELLM_BASEURL", "")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo-1106")
FOLDER_PATH = os.environ.get("AI_FOLDER_PATH", "./ai_files")

# Ensure folder exists
Path(FOLDER_PATH).mkdir(parents=True, exist_ok=True)

# Available functions for the AI
AVAILABLE_FUNCTIONS = {}


def list_files(subfolder: str = "") -> str:
    """List files in the folder or subfolder."""
    target_path = Path(FOLDER_PATH) / subfolder if subfolder else Path(FOLDER_PATH)

    if not target_path.exists():
        return f"Error: Path does not exist: {target_path}"

    if not target_path.is_dir():
        return f"Error: Not a directory: {target_path}"

    try:
        items = list(target_path.iterdir())
        if not items:
            return f"Directory is empty: {target_path}"

        result = f"Files in {target_path}:\n"
        for item in sorted(items):
            item_type = "[DIR] " if item.is_dir() else "[FILE]"
            result += f"  {item_type} {item.name}\n"
        return result.strip()
    except Exception as e:
        return f"Error listing files: {e}"


def read_file(filename: str) -> str:
    """Read the contents of a file."""
    file_path = Path(FOLDER_PATH) / filename

    # Security check: ensure the resolved path is within FOLDER_PATH
    try:
        file_path = file_path.resolve()
        base_path = Path(FOLDER_PATH).resolve()
        if not str(file_path).startswith(str(base_path)):
            return "Error: Access outside the allowed folder is not permitted."
    except Exception:
        return f"Error: Invalid path: {filename}"

    if not file_path.exists():
        return f"Error: File not found: {filename}"

    if not file_path.is_file():
        return f"Error: Not a file: {filename}"

    try:
        content = file_path.read_text(encoding='utf-8')
        return f"Contents of {filename}:\n\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    file_path = Path(FOLDER_PATH) / filename

    # Security check: ensure the resolved path is within FOLDER_PATH
    try:
        file_path = file_path.resolve()
        base_path = Path(FOLDER_PATH).resolve()
        if not str(file_path).startswith(str(base_path)):
            return "Error: Access outside the allowed folder is not permitted."
    except Exception:
        return f"Error: Invalid path: {filename}"

    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"


# Register functions for AI calling
AVAILABLE_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
}

# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the folder. Optionally specify a subfolder path to list files in a subdirectory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subfolder": {
                        "type": "string",
                        "description": "Optional subfolder path to list files in. Leave empty to list the main folder.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Provide the filename or relative path from the main folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename or relative path of the file to read.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, or overwrites it if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename or relative path where to write the content.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are a helpful file agent that helps users manage files.

You have access to three functions:
1. **list_files** - List files in a folder
2. **read_file** - Read the contents of a file
3. **write_file** - Write content to a file

The base folder is: {FOLDER_PATH}

When a user asks you to perform file operations:
- Use list_files to show what files are available
- Use read_file to show contents when asked to read something
- Use write_file to create or modify files

IMPORTANT for multi-step requests:
- If the user's request includes content to write (e.g., "save this information to a file" or the user provides markdown/text content), immediately use write_file with the appropriate filename and content.
- Extract the filename from the user's request (e.g., "save to report.md" -> report.md)
- Write the provided content exactly as given, or summarize/paraphrase if instructed

Only access files within {FOLDER_PATH}. Be helpful and concise.
"""


def call_agent(messages: list, tools: list = None, tool_choice: str = "auto") -> dict:
    """Call the LiteLLM agent with function calling support."""
    if not BASE_URL or not API_KEY:
        return {"error": "LITELLM_BASEURL and LITELLM_API_KEY must be set in environment variables."}

    try:
        response = completion(
            model=MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        return {"error": str(e)}


def execute_tool_calls(tool_calls: list, available_functions: dict) -> list:
    """Execute tool calls and return the results."""
    messages = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name)

        if not function_to_call:
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": f"Error: Function '{function_name}' is not available.",
            })
            continue

        try:
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        except json.JSONDecodeError as e:
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": f"Error: Invalid JSON arguments: {e}",
            })
        except Exception as e:
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": f"Error executing function: {e}",
            })

    return messages


def chat(user_message: str, conversation_history: list = None, max_turns: int = 5) -> tuple[str, list]:
    """Chat with the agent, handling function calls automatically."""
    if conversation_history is None:
        conversation_history = []

    try:
        # Add system prompt and user message
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history + [
            {"role": "user", "content": user_message},
        ]

        # Initial call with tools
        response = call_agent(messages, tools=TOOLS, tool_choice="auto")

        if "error" in response:
            return response["error"], conversation_history

        if not hasattr(response, 'choices') or not response.choices:
            return "Error: Invalid response from API", conversation_history

        response_message = response.choices[0].message
        if not hasattr(response_message, 'tool_calls'):
            # No tool calls, return regular response
            content = response_message.content or ""
            new_history = conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
            ]
            return content, new_history

        tool_calls = response_message.tool_calls

        # If no tool calls, return the regular response
        if not tool_calls:
            content = response_message.content or ""
            new_history = conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
            ]
            return content, new_history

        # Append assistant's message with tool calls
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })

        # Execute tool calls
        tool_results = execute_tool_calls(tool_calls, AVAILABLE_FUNCTIONS)
        messages.extend(tool_results)

        # Get final response from model
        final_response = call_agent(messages, tools=None, tool_choice=None)

        if "error" in final_response:
            return final_response["error"], messages

        if not hasattr(final_response, 'choices') or not final_response.choices:
            return "Error: Invalid final response from API", messages

        final_message = final_response.choices[0].message

        # Update conversation history
        conversation_history = messages + [
            {"role": "assistant", "content": final_message.content},
        ]

        return final_message.content or "", conversation_history

    except Exception as e:
        return f"Error in file_agent: {str(e)}", conversation_history


def main():
    """Interactive CLI for the file agent."""
    print("File Agent with Function Calling")
    print("=" * 50)
    print(f"  LITELLM_BASEURL: {BASE_URL or 'NOT SET'}")
    print(f"  LITELLM_MODEL: {MODEL}")
    print(f"  AI_FOLDER_PATH: {FOLDER_PATH}")
    print()

    # Check if model supports function calling
    try:
        supports = supports_function_calling(model=MODEL)
        print(f"  Function calling supported: {supports}")
    except Exception as e:
        print(f"  Function calling check failed: {e}")

    print("=" * 50)
    print("Type your requests naturally. The AI will call functions as needed.")
    print("Examples:")
    print("  'List all files in the folder'")
    print("  'Read the contents of notes.txt'")
    print("  'Create a new file called hello.txt with Hello World'")
    print("  'Show me what files are in the documents subfolder'")
    print("  'quit' - Exit")
    print()

    conversation_history = []

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            response, conversation_history = chat(user_input, conversation_history)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()