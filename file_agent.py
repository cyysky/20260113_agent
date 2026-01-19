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
import re
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

SYSTEM_PROMPT = f"""You are a helpful file agent that helps users manage files in: {FOLDER_PATH}

You have access to these functions - USE THEM DIRECTLY when needed:
- **list_files** - List files (no arguments needed)
- **read_file** - Read file content (argument: filename)
- **write_file** - Write content (arguments: filename, content)

RULES - Always follow these:
1. If user says "list files", "show files", "what files exist" -> call list_files IMMEDIATELY
2. If user says "read", "show content", "what's in" a file -> call read_file with the filename
3. If user says "write", "create", "save" a file -> call write_file with filename and content
4. If user wants to "read files content" but doesn't specify which -> call list_files first, then read files

When calling functions, output them in this format:
<tool_call>
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

IMPORTANT: Output the tool call format above, not JSON by itself. Do not ask the user for clarification - use the tools directly.

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


def parse_text_tool_calls(content: str) -> list:
    """Parse text-based tool calls like <tool_call>{...}</tool_call>"""
    import re

    # Find all <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    tool_calls = []
    for match in matches:
        try:
            tool_data = json.loads(match.strip())
            name = tool_data.get("name")
            arguments = tool_data.get("arguments", "{}")

            if name and name in AVAILABLE_FUNCTIONS:
                tool_calls.append({
                    "id": f"text_{len(tool_calls)}",
                    "name": name,
                    "arguments": arguments
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return tool_calls


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
        content = response_message.content or ""

        # Check for text-based tool calls (like <tool_call>{...}</tool_call>)
        text_tool_calls = parse_text_tool_calls(content)

        if not hasattr(response_message, 'tool_calls') and not text_tool_calls:
            # No tool calls, return regular response
            new_history = conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
            ]
            return content, new_history

        # Get native tool calls or use text-based ones
        tool_calls = getattr(response_message, 'tool_calls', None) or text_tool_calls

        # If no tool calls, return the regular response
        if not tool_calls:
            new_history = conversation_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
            ]
            return content, new_history

        # Remove text tool calls from content for cleaner output
        cleaned_content = re.sub(r'<?\[?tool_call\]?>\s*\{.*?\}\s*(</tool_call>)?', '', content, flags=re.DOTALL).strip()

        # If using text-based tool calls, execute them directly
        if text_tool_calls and not getattr(response_message, 'tool_calls', None):
            tool_results = []
            for tc in text_tool_calls:
                function_name = tc["name"]
                function_to_call = AVAILABLE_FUNCTIONS.get(function_name)
                if function_to_call:
                    try:
                        args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                        response_text = function_to_call(**args)
                        tool_results.append({
                            "tool_call_id": tc["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": response_text,
                        })
                    except Exception as e:
                        tool_results.append({
                            "tool_call_id": tc["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {e}",
                        })

            messages.extend(tool_results)

            # Get final response from model
            messages.append({"role": "user", "content": "Please provide a helpful response based on the tool results above."})
            final_response = call_agent(messages, tools=None, tool_choice=None)

            if "error" in final_response:
                return final_response["error"], messages

            final_message = final_response.choices[0].message

            conversation_history = messages + [
                {"role": "assistant", "content": final_message.content},
            ]
            return final_message.content or cleaned_content, conversation_history

        # Native tool calls - append and execute
        messages.append({
            "role": "assistant",
            "content": cleaned_content,
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
        final_content = final_message.content or ""

        # Check for text-based tool calls in the final response
        final_text_tool_calls = parse_text_tool_calls(final_content)

        # If there are text-based tool calls, execute them and get another response
        while final_text_tool_calls:
            # Execute text-based tool calls
            for tc in final_text_tool_calls:
                function_name = tc["name"]
                function_to_call = AVAILABLE_FUNCTIONS.get(function_name)
                if function_to_call:
                    try:
                        args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                        response_text = function_to_call(**args)
                        messages.append({
                            "tool_call_id": tc["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": response_text,
                        })
                    except Exception as e:
                        messages.append({
                            "tool_call_id": tc["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {e}",
                        })

            # Get another response from model
            messages.append({"role": "user", "content": "Continue - provide your response based on the tool results."})
            final_response = call_agent(messages, tools=None, tool_choice=None)

            if "error" in final_response:
                return final_response["error"], messages

            final_message = final_response.choices[0].message
            final_content = final_message.content or ""
            final_text_tool_calls = parse_text_tool_calls(final_content)

        # Update conversation history
        conversation_history = messages + [
            {"role": "assistant", "content": final_content},
        ]

        return final_content or "", conversation_history

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