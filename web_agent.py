"""
Web Agent - Browse the internet using LiteLLM with Function Calling

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use (e.g., gpt-4, gpt-3.5-turbo-1106)
- SEARCH_ENGINE: Search engine to use (google_cse or duckduckgo)
- GOOGLE_API_KEY: Google API key for Google Custom Search
- GOOGLE_CSE_ID: Google Custom Search Engine ID

The AI can autonomously call these functions:
- search_web: Search the web for information
- fetch_page: Fetch and extract content from a URL
"""

import os
import sys
import json
import re
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
SEARCH_ENGINE = os.environ.get("SEARCH_ENGINE", "google_cse")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")

# Available functions for the AI
AVAILABLE_FUNCTIONS = {}


def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information using the configured search engine."""
    if SEARCH_ENGINE == "google_cse":
        return search_google_cse(query, num_results)
    else:
        return search_duckduckgo(query, num_results)


def search_google_cse(query: str, num_results: int = 5) -> str:
    """Search using Google Custom Search API."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return "Error: Google API key or CSE ID not configured. Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env"

    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": min(num_results, 10)
        }

        print(f"\n[DEBUG] Searching Google CSE: {query}", file=sys.stderr)

        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()

        data = response.json()
        print(f"[DEBUG] Got {data.get('searchInformation', {}).get('totalResults', 0)} results", file=sys.stderr)

        if "items" not in data or not data["items"]:
            return f"No results found for: {query}"

        results_text = f"Search results for: {query}\n\n"
        for i, item in enumerate(data["items"][:num_results], 1):
            title = item.get("title", "No title")
            link = item.get("link", "No URL")
            snippet = item.get("snippet", "")
            results_text += f"{i}. {title}\n   URL: {link}\n   {snippet[:200]}...\n\n"
            print(f"[DEBUG] {i}. {title[:50]}... -> {link[:60]}...", file=sys.stderr)

        results_text += f"({len(data['items'])} results)"
        return results_text

    except httpx.HTTPStatusError as e:
        print(f"[DEBUG] HTTP Error: {e.response.status_code}", file=sys.stderr)
        return f"Error: Google API request failed (HTTP {e.response.status_code}). Check your API key and CSE ID."
    except Exception as e:
        print(f"[DEBUG] Exception: {e}", file=sys.stderr)
        return f"Error searching with Google: {e}"


def search_duckduckgo(query: str, num_results: int = 5) -> str:
    """Search using DuckDuckGo HTML (no API key required)."""
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query, "kl": "us-en"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()

        # Parse results from HTML using regex
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
        return f"Error searching with DuckDuckGo: {e}"


def fetch_page(url: str, extract_text: bool = True) -> str:
    """Fetch and extract content from a URL."""
    try:
        print(f"\n[DEBUG] Fetching: {url}", file=sys.stderr)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        print(f"[DEBUG] Status: {response.status_code}, Size: {len(response.text)} bytes", file=sys.stderr)

        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type:
            # Simple HTML to text extraction
            text = response.text
            original_len = len(text)

            # Remove script and style elements
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML comments
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

            # Remove navigation and footer elements
            text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Get title if available
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', response.text, re.IGNORECASE)
            title = title_match.group(1) if title_match else "No title"

            # Clean title
            title = re.sub(r'\s+', ' ', title).strip()

            print(f"[DEBUG] Title: {title}, Cleaned: {original_len} -> {len(text)} chars", file=sys.stderr)

            # Return first 4000 chars
            preview = text[:4000]
            result = f"Title: {title}\nURL: {url}\n\nContent:\n{preview}"

            if len(text) > 4000:
                result += f"\n\n... (truncated, {len(text)} total characters)"

            # Show first 500 chars of content for debug
            print(f"[DEBUG] Content preview: {text[:500]}...", file=sys.stderr)

            return result
        else:
            print(f"[DEBUG] Non-HTML content: {content_type}", file=sys.stderr)
            return f"URL: {url}\nContent-Type: {content_type}\n\n[Binary content - not displayable as text]"

    except httpx.HTTPStatusError as e:
        print(f"[DEBUG] HTTP Error: {e.response.status_code}", file=sys.stderr)
        return f"Error fetching URL (HTTP {e.response.status_code}): {url}"
    except Exception as e:
        print(f"[DEBUG] Exception: {e}", file=sys.stderr)
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


def parse_tool_calls_from_content(content: str) -> list:
    """Parse tool calls embedded in response content (for models that output XML-style tags)."""
    tool_calls = []

    print(f"\n[DEBUG] Parsing content for tool calls (len={len(content)})", file=sys.stderr)
    print(f"[DEBUG] Content preview:\n{content[:500]}", file=sys.stderr)

    # Match <tool_call>...</tool_call> blocks with JSON inside
    pattern = r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>'
    matches = re.findall(pattern, content)

    print(f"[DEBUG] Found {len(matches)} matches", file=sys.stderr)

    for i, match in enumerate(matches):
        print(f"[DEBUG] Match {i}: {match[:100]}...", file=sys.stderr)
        try:
            func_data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": func_data.get("name", ""),
                    "arguments": json.dumps(func_data.get("arguments", {}))
                }
            })
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {e}", file=sys.stderr)
            continue

    return tool_calls


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
        # Handle both dict and object formats
        if isinstance(tool_call, dict):
            function_name = tool_call.get("function", {}).get("name", "")
            function_args_str = tool_call.get("function", {}).get("arguments", "{}")
            tool_call_id = tool_call.get("id", "unknown")
        else:
            function_name = tool_call.function.name
            function_args_str = tool_call.function.arguments
            tool_call_id = getattr(tool_call, 'id', "unknown")

        function_to_call = available_functions.get(function_name)

        if not function_to_call:
            messages.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": f"Error: Function '{function_name}' is not available.",
            })
            continue

        try:
            function_args = json.loads(function_args_str)
            function_response = function_to_call(**function_args)

            messages.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        except json.JSONDecodeError as e:
            messages.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "content": f"Error: Invalid JSON arguments: {e}",
            })
        except Exception as e:
            messages.append({
                "tool_call_id": tool_call_id,
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

        # Process response and handle any tool calls (may be multiple rounds)
        turn_count = 0
        while turn_count < max_turns:
            try:
                response_message = response.choices[0].message
            except (AttributeError, IndexError):
                return "Error: Invalid response message", messages

            content = response_message.content or ""

            # Try standard tool_calls first, then fall back to parsing from content
            tool_calls = getattr(response_message, 'tool_calls', None)

            # If no tool_calls in response, check if they are embedded in content as XML tags
            if not tool_calls:
                tool_calls = parse_tool_calls_from_content(content)

            # If no tool calls found, we're done
            if not tool_calls:
                conversation_history = messages + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": content},
                ]
                return content, conversation_history

            # Remove the tool call XML tags from the content for display
            clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
            if not clean_content:
                clean_content = None  # Let it be None if no other content

            # Append assistant's message with tool calls
            messages.append({
                "role": "assistant",
                "content": clean_content,
                "tool_calls": [
                    {
                        "id": tc.get("id", f"call_{i}") if isinstance(tc, dict) else tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name", "") if isinstance(tc, dict) else tc.function.name,
                            "arguments": tc.get("function", {}).get("arguments", "{}") if isinstance(tc, dict) else tc.function.arguments
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            })

            # Execute tool calls
            tool_results = execute_tool_calls(tool_calls, AVAILABLE_FUNCTIONS)
            messages.extend(tool_results)

            # Get next response from model (keep tools enabled for multi-round calls)
            response = call_agent(messages, tools=TOOLS, tool_choice="auto")

            if "error" in response:
                return response["error"], messages

            if not hasattr(response, 'choices') or not response.choices:
                return "Error: Invalid response in loop", messages

            turn_count += 1

        # Max turns reached, return last response
        try:
            response_message = response.choices[0].message
            content = response_message.content or ""
        except (AttributeError, IndexError):
            content = "Max turns reached"
        conversation_history = messages + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": content},
        ]
        return content, conversation_history

    except Exception as e:
        return f"Error in web_agent: {str(e)}", conversation_history


def main():
    """Interactive CLI for the web agent."""
    print("Web Agent with Function Calling")
    print("=" * 50)
    print(f"  LITELLM_BASEURL: {BASE_URL or 'NOT SET'}")
    print(f"  LITELLM_MODEL: {MODEL}")
    print(f"  SEARCH_ENGINE: {SEARCH_ENGINE}")
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