"""
Summary Agent - Summarizes search results and web content into well-formatted markdown

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use (e.g., gpt-4, gpt-3.5-turbo-1106)

This agent takes raw search/fetch results and creates a clean markdown summary.
"""

import os
import json
from typing import Dict, List, Callable, Optional
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
BASE_URL = os.environ.get("LITELLM_BASEURL", "")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo-1106")

SYSTEM_PROMPT = """You are a professional content summarizer. Your job is to take raw search results and web content, and create a well-organized, comprehensive markdown document.

Guidelines:
1. Extract key information and organize it logically
2. Use proper markdown formatting (headers, lists, links)
3. Include source URLs where appropriate
4. Summarize long content while preserving key facts
5. Create a cohesive narrative from multiple sources
6. Be factual and cite sources
7. Format with good structure - introduction, sections, conclusion

Output format should be clean markdown without code blocks (```markdown ... ```).
"""


def call_agent(messages: list, tools: list = None, tool_choice: str = None) -> dict:
    """Call the LiteLLM agent."""
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


def summarize(
    topic: str,
    raw_content: str,
    conversation_history: Optional[List[Dict]] = None,
    max_turns: int = 5
) -> tuple[str, List[Dict]]:
    """Summarize raw content into a well-formatted markdown document.

    Args:
        topic: The topic/title of the document
        raw_content: The raw search/fetch results to summarize
        conversation_history: Optional conversation history
        max_turns: Maximum turns (kept for API compatibility)

    Returns:
        tuple: (summarized_content, conversation_history)
    """
    if conversation_history is None:
        conversation_history = []

    # Create a prompt that instructs the AI to create a comprehensive summary
    summary_prompt = f"""Topic: {topic}

Raw Research Data:
{raw_content}

Task: Create a comprehensive, well-formatted markdown summary about this topic.
Include:
1. A clear title
2. Sections organized by themes/topics
3. Key facts and statistics
4. Source citations (use the URLs from the content)
5. Clean formatting with headers, bullet points, and links

Write directly in markdown format (no code blocks). Make it comprehensive and informative.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ] + conversation_history + [
        {"role": "user", "content": summary_prompt},
    ]

    response = call_agent(messages, tools=None, tool_choice=None)

    if "error" in response:
        return response["error"], conversation_history

    try:
        content = response.choices[0].message.content or ""
        new_history = conversation_history + [
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": content},
        ]
        return content, new_history
    except (AttributeError, IndexError):
        return "Error: Invalid response from API", conversation_history


def chat(user_message: str, conversation_history: list = None, max_turns: int = 5) -> tuple[str, list]:
    """Chat with the summary agent."""
    if conversation_history is None:
        conversation_history = []

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history + [
            {"role": "user", "content": user_message},
        ]

        response = call_agent(messages, tools=None, tool_choice=None)

        if "error" in response:
            return response["error"], conversation_history

        content = response.choices[0].message.content or ""
        new_history = conversation_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": content},
        ]
        return content, new_history

    except Exception as e:
        return f"Error in summary_agent: {str(e)}", conversation_history


if __name__ == "__main__":
    # Simple test
    print("Summary Agent")
    print("=" * 50)
    print(f"LITELLM_BASEURL: {BASE_URL or 'NOT SET'}")
    print(f"LITELLM_MODEL: {MODEL}")
    print()

    topic = input("Enter topic: ")
    raw = input("Paste raw content (or press Enter for test): ")

    if not raw.strip():
        raw = """Search results for: Malaysia AI news

1. e-Conomy SEA 2025: Malaysia takes 32% of regional AI funding
   URL: https://example.com/news/malaysia-ai-funding
   Malaysia captured 32% of Southeast Asia's AI funding.

2. ASEAN AI Malaysia Summit 2025
   URL: https://example.com/asean-summit
   Summit scheduled for August 2025 in Kuala Lumpur."""

    result, _ = summarize(topic, raw)
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(result)