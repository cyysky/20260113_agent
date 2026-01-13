"""
Orchestrator Agent - Manages multiple agents and routes requests

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use (e.g., gpt-4, gpt-3.5-turbo-1106)

Usage:
    python orchestrator.py
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
BASE_URL = os.environ.get("LITELLM_BASEURL", "")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo-1106")


class Agent:
    """Represents a registered agent with its capabilities."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: List[Dict],
        available_functions: Dict[str, Callable],
        chat_func: Callable[[str, list, int], tuple[str, list]]
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        self.available_functions = available_functions
        self.chat_func = chat_func


class Orchestrator:
    """Orchestrates multiple agents and routes requests appropriately."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.conversation_history: List[Dict] = []

    def register_agent(self, agent: Agent):
        """Register a new agent."""
        self.agents[agent.name] = agent
        print(f"[Orchestrator] Registered agent: {agent.name}")

    def list_agents(self) -> str:
        """List all registered agents."""
        if not self.agents:
            return "No agents registered."

        result = "Available Agents:\n"
        for name, agent in self.agents.items():
            result += f"  - {name}: {agent.description}\n"
        return result.strip()

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get all agent capabilities for routing decisions."""
        capabilities = {}
        for name, agent in self.agents.items():
            capabilities[name] = {
                "description": agent.description,
                "tools": [t.get("function", {}).get("name") for t in agent.tools],
            }
        return capabilities

    def route_request(self, user_message: str) -> tuple[str, list, bool]:
        """Route the user request to the appropriate agent(s).

        Returns:
            tuple: (primary_agent, list_of_agents, requires_sequential)
        """
        if not self.agents:
            return "", [], False

        # Build a routing prompt
        capabilities = self.get_agent_capabilities()
        routing_prompt = f"""You are a request router. Your job is to determine which agent(s) should handle the user's request.

Available Agents and their capabilities:
{json.dumps(capabilities, indent=2)}

User request: "{user_message}"

Respond with a JSON object containing:
- "agents": array of agent names to use in order
- "reason": brief explanation of why

IMPORTANT: For research + save tasks (like "search for X and save to file"), use this order:
["web_agent", "summary_agent", "file_agent"]
1. web_agent - gathers information
2. summary_agent - creates well-formatted summary from raw results
3. file_agent - writes final content to file

Example responses:
- "{{"agents": ["web_agent", "summary_agent", "file_agent"], "reason": "Research topic then summarize and save to file"}}"
- "{{"agents": ["file_agent"], "reason": "Simple file operation"}}"
- "{{"agents": ["web_agent"], "reason": "Web research only"}}"
"""

        try:
            response = completion(
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                messages=[{"role": "system", "content": routing_prompt}],
                tool_choice=None,
            )

            content = response.choices[0].message.content or ""
            # Try to parse the JSON from the response
            try:
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                routing = json.loads(content)
                agent_names = routing.get("agents", [])
                reason = routing.get("reason", "")

                # Filter to only registered agents
                valid_agents = [a for a in agent_names if a in self.agents]

                if valid_agents:
                    return valid_agents[0], valid_agents, len(valid_agents) > 1
                else:
                    # Fallback to first agent
                    return list(self.agents.keys())[0], [list(self.agents.keys())[0]], False
            except (json.JSONDecodeError, IndexError):
                # Fallback: simple keyword matching
                return self._simple_route(user_message)

        except Exception as e:
            return self._simple_route(user_message)

    def _simple_route(self, user_message: str) -> tuple:
        """Simple keyword-based routing as fallback."""
        message_lower = user_message.lower()
        agents_to_use = []

        # Check for search/fetch web operations -> web_agent
        if any(kw in message_lower for kw in ["search", "find", "fetch", "browse", "web", "internet", "look up"]):
            agents_to_use.append("web_agent")

        # Check for research + save operations -> summary_agent + file_agent
        if any(kw in message_lower for kw in ["save", "write", "create"]) and "search" in message_lower:
            # For search + save, add summary and file agents
            if "summary_agent" in self.agents:
                agents_to_use.append("summary_agent")
            if "file_agent" in self.agents:
                agents_to_use.append("file_agent")
        elif any(kw in message_lower for kw in ["write", "save", "create", "file"]):
            agents_to_use.append("file_agent")

        # If we have matches, use them (dedup and preserve order)
        if agents_to_use:
            agents_to_use = list(dict.fromkeys(agents_to_use))
            return agents_to_use[0], agents_to_use, len(agents_to_use) > 1

        # Default to first registered agent
        if self.agents:
            default = list(self.agents.keys())[0]
            return default, [default], False

        return "", [], False

    def process_request(self, user_message: str) -> str:
        """Process a user request by routing to the appropriate agent(s)."""
        if not self.agents:
            return "Error: No agents registered."

        # Route the request
        primary_agent, all_agents, requires_sequential = self.route_request(user_message)

        if not primary_agent:
            return "Error: Could not determine which agent to use."

        if requires_sequential and len(all_agents) > 1:
            # Multi-step request: execute agents in sequence
            return self._process_sequential(user_message, all_agents)
        else:
            # Single agent request
            agent = self.agents[primary_agent]
            print(f"\n[Routing] {primary_agent}")
            print("-" * 50)

            response, self.conversation_history = agent.chat_func(
                user_message, self.conversation_history, max_turns=3
            )
            return response

    def _extract_tool_results_from_history(self, conversation_history: list) -> str:
        """Extract full tool call results from conversation history (non-truncated)."""
        results = []
        for msg in conversation_history:
            if msg.get("role") == "tool":
                name = msg.get("name", "")
                content = msg.get("content", "")
                if name in ["search_web", "fetch_page"]:
                    results.append(f"=== {name.upper()} RESULT ===\n{content}")
        return "\n\n".join(results)

    def _process_sequential(self, user_message: str, agent_names: list) -> str:
        """Process a request using multiple agents in sequence."""
        accumulated_content = ""
        tool_results = ""

        for i, agent_name in enumerate(agent_names):
            agent = self.agents[agent_name]
            print(f"\n[Step {i+1}/{len(agent_names)}] {agent_name}")
            print("-" * 50)

            if i == 0:
                # First agent runs normally - give it more turns for web search
                max_turns_for_agent = 15 if agent_name == "web_agent" else 3
                response, self.conversation_history = agent.chat_func(
                    user_message, self.conversation_history, max_turns=max_turns_for_agent
                )
                accumulated_content = response
                # Extract the actual tool results from history (full, non-truncated content)
                tool_results = self._extract_tool_results_from_history(self.conversation_history)

            elif agent_name == "summary_agent":
                # Use summary agent to create well-formatted summary
                print("\nSummarizing raw content...")
                try:
                    import summary_agent
                    # Extract topic from user message
                    import re
                    topic_match = re.search(r'(?:search|find|look up|get)\s+(?:for\s+)?(.+?)\s*(?:and|then|to|into|\.)', user_message, re.IGNORECASE)
                    if not topic_match:
                        topic_match = re.search(r'(?:about|on|regarding)\s+(.+?)(?:\s+and|\s+to|\s+into|\s*$)', user_message, re.IGNORECASE)
                    topic = topic_match.group(1).strip() if topic_match else "Research Summary"

                    summarized, self.conversation_history = summary_agent.summarize(
                        topic, tool_results, self.conversation_history
                    )
                    accumulated_content = summarized
                    print(f"\nSummary created ({len(summarized)} chars)")
                except Exception as e:
                    accumulated_content = f"# Summary\n\n{tool_results}"
                    print(f"\nError in summary_agent: {e}")
                    # Print first 500 chars of what would be written
                    preview = accumulated_content[:500] + "..." if len(accumulated_content) > 500 else accumulated_content
                    print(f"Writing raw content instead:\n{preview}\n")

            elif agent_name == "file_agent":
                # Pass the content to write
                print(f"\nPassing content to {agent_name}...")
                import re
                filename_match = re.search(r'save.*?into\s+(\S+\.md)', user_message, re.IGNORECASE)
                if not filename_match:
                    filename_match = re.search(r'save.*?to\s+(\S+\.md)', user_message, re.IGNORECASE)
                filename = filename_match.group(1) if filename_match else "output.md"

                content_to_write = accumulated_content

                # Check if content has markdown code blocks
                if "```markdown" in content_to_write:
                    content_to_write = content_to_write.split("```markdown")[1].split("```")[0]
                elif "```" in content_to_write:
                    code_match = re.search(r'```[a-z]*\n([\s\S]*?)\n```', content_to_write)
                    if code_match:
                        content_to_write = code_match.group(1)

                # Call file_agent's write_file directly
                result = agent.available_functions.get("write_file")(filename, content_to_write.strip())
                print(f"\n{result}")
                accumulated_content = result

            else:
                # For other agents, pass context
                context_prompt = f"""Original request: {user_message}

Information gathered so far:
{accumulated_content}

Complete the task."""
                response, self.conversation_history = agent.chat_func(
                    context_prompt, self.conversation_history, max_turns=3
                )
                accumulated_content = response

            print(f"\n{accumulated_content}\n")

        return accumulated_content

    def run_loop(self):
        """Run the main interactive loop."""
        print("Orchestrator Agent - Multi-Agent System")
        print("=" * 50)
        print(f"  LITELLM_BASEURL: {BASE_URL or 'NOT SET'}")
        print(f"  LITELLM_MODEL: {MODEL}")
        print(f"  Registered Agents: {len(self.agents)}")
        print()
        print(self.list_agents())
        print()
        print("=" * 50)
        print("Commands:")
        print("  /agents - List all registered agents")
        print("  /history - Show conversation history")
        print("  /clear - Clear conversation history")
        print("  /quit - Exit")
        print("=" * 50)
        print()

        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    print("Goodbye!")
                    break

                if user_input.lower() == "/agents":
                    print(f"\n{self.list_agents()}\n")
                    continue

                if user_input.lower() == "/history":
                    print(f"\nConversation history ({len(self.conversation_history)} messages):")
                    for msg in self.conversation_history[-10:]:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if len(content) > 200:
                            content = content[:200] + "..."
                        print(f"  [{role}]: {content}")
                    print()
                    continue

                if user_input.lower() == "/clear":
                    self.conversation_history = []
                    print("Conversation history cleared.\n")
                    continue

                # Process the request
                response = self.process_request(user_input)
                print(f"\n{response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# Factory functions for creating agents from existing modules

def create_file_agent():
    """Create a file agent from file_agent module."""
    import file_agent

    return Agent(
        name="file_agent",
        description="File operations - list, read, and write files",
        system_prompt=file_agent.SYSTEM_PROMPT,
        tools=file_agent.TOOLS,
        available_functions=file_agent.AVAILABLE_FUNCTIONS,
        chat_func=file_agent.chat,
    )


def create_web_agent():
    """Create a web agent from web_agent module."""
    import web_agent

    return Agent(
        name="web_agent",
        description="Web browsing - search the web and fetch pages",
        system_prompt=web_agent.SYSTEM_PROMPT,
        tools=web_agent.TOOLS,
        available_functions=web_agent.AVAILABLE_FUNCTIONS,
        chat_func=web_agent.chat,
    )


def create_summary_agent():
    """Create a summary agent from summary_agent module."""
    import summary_agent

    return Agent(
        name="summary_agent",
        description="Content summarizer - turns raw research into well-formatted markdown",
        system_prompt=summary_agent.SYSTEM_PROMPT,
        tools=[],
        available_functions={},
        chat_func=summary_agent.chat,
    )


def main():
    """Main entry point for the orchestrator."""
    orchestrator = Orchestrator()

    # Register default agents
    orchestrator.register_agent(create_file_agent())
    orchestrator.register_agent(create_web_agent())
    orchestrator.register_agent(create_summary_agent())

    # Run the interactive loop
    orchestrator.run_loop()


if __name__ == "__main__":
    main()