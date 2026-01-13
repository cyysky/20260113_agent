"""
Web Agent - Browse the internet using LiteLLM with Function Calling

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use (e.g., gpt-4, gpt-3.5-turbo-1106)

The AI can autonomously call these functions:
- search_web: Search the web for information
- fetch_page: Fetch and extract content from a URL
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv
from litellm import completion, supports_function_calling
import httpx

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
BASE_URL = os.environ.get("LITELLM_BASEURL", "")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo-1106")

# Available functions for the AI
AVAILABLE_FUNCTIONS = {}


def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information using a search engine."""
    try:
        # Use DuckDuckGo HTML search (no API key required)
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query, "kl": "us-en"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()

        # Parse results from HTML
        results = []
        lines = response.text.split('\n')
        in_result = False
        current_result = {}

        for line in lines:
            if 'result__a' in line or 'result__title' in line:
                in_result = True
                current_result = {}
            elif in_result and '<a href="' in line:
                start = line.find('<a href="') + len('<a href="')
                end = line.find('"', start)
                if start > 9 and end > start:
                    current_result['url'] = line[start:end]
            elif in_result and '>' in line and not line.strip().startswith('<'):
                text = line.strip()
                if text and len(text) > 10:
                    current_result['title'] = text[:200]
            elif in_result and 'result__snippet' in line:
                in_result = True  # Keep collecting snippet

        # Extract cleaner results using simple parsing
        import re
        pattern = r'<a href="([^"]+)".*?>([^<]+)</a>'
        matches = re.findall(pattern, response.text)

        results_text = f"Search results for: {query}\n\n"
        count = 0
        seen_urls = set()

        for url, title in matches:
            if count >= num_results:
                break
            if url.startswith('http') and url not in seen_urls:
                seen_urls.add(url)
                clean_title = title.replace('<b>', '').replace('</b>', '')[:100]
                results_text += f"{count + 1}. {clean_title}\n   URL: {url}\n\n"
                count += 1

        if count == 0:
            return f"No results found for: {query}"

        results_text += f"({count} results)"
        return results_text

    except Exception as e:
        return f"Error searching web: {e}"


def fetch_page(url: str, extract_text: bool = True) -> str:
    """Fetch and extract content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type:
            # Simple HTML to text extraction
            import re
            text = response.text

            # Remove script and style elements
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Get title if available
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text, re.IGNORECASE)
            title = title_match.group(1) if title_match else "No title"

            # Return first 3000 chars
            preview = text[:3000]
            result = f"Title: {title}\nURL: {url}\n\nContent:\n{preview}"

            if len(text) > 3000:
                result += f"\n\n... (truncated, {len(text)} total characters)"

            return result
        else:
            return f"URL: {url}\nContent-Type: {content_type}\n\n[Binary content - not displayable as text]"

    except httpx.HTTPStatusError as e:
        return f"Error fetching URL (HTTP {e.response.status_code}): {url}"
    except Exception as e:
        return f"Error fetching URL: {e}"


# Register functions for AI calling
AVAILABLE_FUNCTIONS = {
    "search_web": search_web,
    "fetch_page": fetch_page,
}

# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information. Use this when you need to find up-to-date information, facts, or sources on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and use keywords.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch and extract text content from a URL. Use this to get detailed information from a specific webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from.",
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Whether to extract text from HTML (default: true)",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful web browsing assistant.

You have access to two functions:
1. **search_web** - Search the web for information
2. **fetch_page** - Fetch and extract content from a URL

When a user asks you to find information:
- Use search_web to find relevant sources
- Use fetch_page to get detailed content from promising URLs
- Summarize and present the information clearly

Be thorough and verify information across multiple sources when appropriate. Always cite the sources you found."""


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


def chat(user_message: str, conversation_history: list = None) -> tuple[str, list]:
    """Chat with the agent, handling function calls automatically."""
    if conversation_history is None:
        conversation_history = []

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

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # If no tool calls, return the regular response
    if not tool_calls:
        return response_message.content or "", conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_message.content},
        ]

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

    final_message = final_response.choices[0].message

    # Update conversation history
    conversation_history = messages + [
        {"role": "assistant", "content": final_message.content},
    ]

    return final_message.content or "", conversation_history


def main():
    """Interactive CLI for the web agent."""
    print("Web Agent with Function Calling")
    print("=" * 50)
    print(f"  LITELLM_BASEURL: {BASE_URL or 'NOT SET'}")
    print(f"  LITELLM_MODEL: {MODEL}")
    print()

    # Check if model supports function calling
    try:
        supports = supports_function_calling(model=MODEL)
        print(f"  Function calling supported: {supports}")
    except Exception as e:
        print(f"  Function calling check failed: {e}")

    print("=" * 50)
    print("Ask me anything - I'll search the web and fetch pages as needed.")
    print("Examples:")
    print("  'What is the latest news about AI?'")
    print("  'Find documentation for Python requests library'")
    print("  'What are the top-rated restaurants in New York?'")
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