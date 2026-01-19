"""
Orchestrator Agent - Manages multiple agents with LLM-planned workflows

Configure via environment variables:
- LITELLM_BASEURL: The base URL for LiteLLM API
- LITELLM_API_KEY: The API key for authentication
- LITELLM_MODEL: The model to use (e.g., gpt-4, gpt-3.5-turbo-1106)

Usage:
    python orchestrator.py
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Callable, Union
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


class AgentStep:
    """Represents a step in an agent execution plan."""

    def __init__(self, agent_name: str, purpose: str, max_turns: int = 3, input_type: str = "user_message"):
        self.agent_name = agent_name
        self.purpose = purpose
        self.max_turns = max_turns
        self.input_type = input_type  # "user_message", "accumulated", "tool_results"

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "purpose": self.purpose,
            "max_turns": self.max_turns,
            "input": self.input_type,
        }


class Orchestrator:
    """Orchestrates multiple agents with LLM-planned workflows."""

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
        """Get all agent capabilities for planning decisions."""
        capabilities = {}
        for name, agent in self.agents.items():
            capabilities[name] = {
                "description": agent.description,
                "tools": [t.get("function", {}).get("name") for t in agent.tools if t.get("function")],
            }
        return capabilities

    def plan_execution(self, user_message: str) -> List[AgentStep]:
        """Let the LLM plan the execution pipeline.

        Returns:
            List of AgentStep objects representing the execution plan
        """
        if not self.agents:
            return []

        capabilities = self.get_agent_capabilities()

        planning_prompt = f"""You are an expert workflow planner. Given a user's request, create a detailed execution plan using the available agents.

Available Agents:
{json.dumps(capabilities, indent=2)}

User Request: "{user_message}"

Create a step-by-step execution plan. Respond with a JSON object containing:
- "steps": array of execution steps

Each step should have:
- "agent": the agent name to use
- "purpose": what this step accomplishes
- "max_turns": how many turns to allow (higher for research tasks)
- "input": what to pass ("user_message", "accumulated", "tool_results", or "step_output")

Guidelines:
1. For file operations (list/read/write): use file_agent in a single step - it can handle multi-turn tool execution internally
2. For web research: web_agent needs 10-15 turns to search and fetch multiple sources
3. For summarization: pass the accumulated content or tool results
4. For research + save tasks: use web_agent first, then summary_agent, then file_agent
5. Keep pipelines minimal - don't add unnecessary steps

Example Response:
{{
    "steps": [
        {{
            "agent": "web_agent",
            "purpose": "Research the topic and gather information from the web",
            "max_turns": 15,
            "input": "user_message"
        }},
        {{
            "agent": "summary_agent",
            "purpose": "Create a well-formatted summary from the research",
            "max_turns": 3,
            "input": "tool_results"
        }},
        {{
            "agent": "file_agent",
            "purpose": "Write the summary to the specified file",
            "max_turns": 3,
            "input": "step_output"
        }}
    ],
    "reason": "The user wants to research a topic and save results to a file"
}}
"""

        try:
            response = completion(
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                messages=[{"role": "system", "content": planning_prompt}],
                tool_choice=None,
            )

            content = response.choices[0].message.content or ""

            # Parse the JSON response
            try:
                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                plan_data = json.loads(content)
                steps = plan_data.get("steps", [])

                # If LLM returned empty steps, it's casual conversation
                if not steps:
                    return None

                # Convert to AgentStep objects, filtering valid agents
                agent_steps = []
                for step in steps:
                    agent_name = step.get("agent")
                    if agent_name in self.agents:
                        agent_step = AgentStep(
                            agent_name=agent_name,
                            purpose=step.get("purpose", ""),
                            max_turns=step.get("max_turns", 10),
                            input_type=step.get("input", "user_message"),
                        )
                        agent_steps.append(agent_step)

                if agent_steps:
                    print(f"[Planning] LLM planned {len(agent_steps)} steps")
                    for i, step in enumerate(agent_steps):
                        print(f"  {i+1}. {step.agent_name}: {step.purpose}")
                    return agent_steps

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"[Planning] JSON parse error: {e}")

        except Exception as e:
            print(f"[Planning] LLM error: {e}")

        # Fallback to simple planning
        print("[Planning] Using fallback planner")
        return self._fallback_plan(user_message)

    def _fallback_plan(self, user_message: str) -> Union[List[AgentStep], None]:
        """Simple fallback planner based on keywords. Returns None for casual conversation."""
        message_lower = user_message.lower()
        steps = []

        # Check for web research
        if any(kw in message_lower for kw in ["search", "find", "browse", "web", "look up", "research"]):
            steps.append(AgentStep(
                agent_name="web_agent",
                purpose="Search and gather information from the web",
                max_turns=15,
                input_type="user_message"
            ))

        # Check for summarization need
        if any(kw in message_lower for kw in ["summarize", "summary", "summarise"]):
            steps.append(AgentStep(
                agent_name="summary_agent",
                purpose="Create a well-formatted summary",
                max_turns=3,
                input_type="accumulated"
            ))

        # Check for file operations
        if any(kw in message_lower for kw in ["save", "write", "create", "file", "export"]):
            steps.append(AgentStep(
                agent_name="file_agent",
                purpose="Write content to file",
                max_turns=3,
                input_type="accumulated"
            ))

        # If only file operation without research
        if not steps and any(kw in message_lower for kw in ["list", "read", "show files"]):
            steps.append(AgentStep(
                agent_name="file_agent",
                purpose="Perform file operation",
                max_turns=10,
                input_type="user_message"
            ))

        return steps if steps else None

    def _extract_tool_results_from_history(self, conversation_history: list) -> str:
        """Extract full tool call results from conversation history."""
        results = []
        for msg in conversation_history:
            if msg.get("role") == "tool":
                name = msg.get("name", "")
                content = msg.get("content", "")
                if name in ["search_web", "fetch_page"]:
                    results.append(f"=== {name.upper()} RESULT ===\n{content}")
        return "\n\n".join(results)

    def _execute_plan(self, user_message: str, plan: List[AgentStep]) -> str:
        """Execute the planned workflow."""
        accumulated_content = ""
        step_outputs = {}  # Store outputs from each step
        tool_results = ""

        for i, step in enumerate(plan):
            agent = self.agents.get(step.agent_name)
            if not agent:
                print(f"[Warning] Agent {step.agent_name} not found, skipping")
                continue

            print(f"\n[Step {i+1}/{len(plan)}] {step.agent_name}")
            print(f"Purpose: {step.purpose}")
            print("-" * 50)

            # Determine input for this step
            if step.input_type == "user_message":
                # If there's accumulated content from previous steps, include it
                if accumulated_content and i > 0:
                    current_input = f"""Original Request: {user_message}

Previous Step Results:
{accumulated_content}

Continue by completing the current step: {step.purpose}"""
                else:
                    current_input = user_message
            elif step.input_type == "accumulated":
                current_input = f"""Original Request: {user_message}

Previous Results:
{accumulated_content}

Continue based on the above context."""
            elif step.input_type == "tool_results":
                current_input = f"""Research Topic: {user_message}

Raw Research Data:
{tool_results}

Task: Create a comprehensive, well-formatted summary."""
            elif step.input_type == "step_output":
                # Get output from a specific previous step
                prev_step_idx = -1
                for j in range(i - 1, -1, -1):
                    if plan[j].agent_name in step_outputs:
                        prev_step_idx = j
                        break
                if prev_step_idx >= 0:
                    current_input = f"""Write the following content to a file:

{step_outputs[plan[prev_step_idx].agent_name]}

Original request: {user_message}"""
                else:
                    current_input = f"""Write the following content:

{accumulated_content}"""
            else:
                current_input = user_message

            # Execute the agent
            try:
                response, self.conversation_history = agent.chat_func(
                    current_input, self.conversation_history, max_turns=step.max_turns
                )
                accumulated_content = response
                step_outputs[step.agent_name] = response

                # Extract tool results if this is web_agent
                if step.agent_name == "web_agent":
                    tool_results = self._extract_tool_results_from_history(self.conversation_history)

                print(f"\n[Output ({len(response)} chars)]\n{response[:500]}..." if len(response) > 500 else f"\n[Output]\n{response}")

            except Exception as e:
                print(f"[Error] Step failed: {e}")
                accumulated_content += f"\n[Error in {step.agent_name}]: {str(e)}"

        return accumulated_content

    def process_request(self, user_message: str) -> str:
        """Process a user request by planning and executing a workflow."""
        if not self.agents:
            return "Error: No agents registered."

        print(f"\n[Request] {user_message}")

        # Let LLM plan the execution
        plan = self.plan_execution(user_message)

        if plan is None:
            # No execution needed - casual conversation
            return self._handle_casual_conversation(user_message)

        if not plan:
            return "Error: Could not create execution plan."

        # Execute the plan
        result = self._execute_plan(user_message, plan)
        return result

    def _handle_casual_conversation(self, user_message: str) -> str:
        """Handle casual conversation using the LLM."""
        casual_prompt = f"""You are a friendly orchestrator assistant. The user just said: "{user_message}"

This appears to be casual conversation or a greeting. Respond in a friendly, helpful way.
Mention you can help with file operations, web research, and summarization.

Keep it concise and conversational. Plain text only, no special characters or emojis."""

        try:
            response = completion(
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                messages=[{"role": "system", "content": casual_prompt}],
            )
            content = response.choices[0].message.content or "Hello! How can I help you today?"
            # Remove any problematic characters
            return content.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            return "Hello! I'm an orchestrator for file, web, and summary agents. I can help you research topics, manage files, and create summaries. What would you like to do?"

    def run_loop(self):
        """Run the main interactive loop."""
        print("Orchestrator Agent - LLM-Planned Multi-Agent System")
        print("=" * 60)
        print(f"  LITELLM_BASEURL: {BASE_URL or 'NOT SET'}")
        print(f"  LITELLM_MODEL: {MODEL}")
        print(f"  Registered Agents: {len(self.agents)}")
        print()
        print(self.list_agents())
        print()
        print("=" * 60)
        print("Commands:")
        print("  /agents - List all registered agents")
        print("  /plan <query> - Show the planned agent pipeline for a query")
        print("  /history - Show conversation history")
        print("  /clear - Clear conversation history")
        print("  /quit - Exit")
        print("=" * 60)
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

                if user_input.lower() == "/plan":
                    print("[Error] Please provide a query. Usage: /plan <query>")
                    continue

                if user_input.lower().startswith("/plan "):
                    query = user_input[6:].strip()
                    plan = self.plan_execution(query)
                    print(f"\nPlanned Pipeline ({len(plan)} steps):")
                    for i, step in enumerate(plan):
                        print(f"  {i+1}. {step.agent_name}")
                        print(f"     - {step.purpose}")
                        print(f"     - max_turns: {step.max_turns}, input: {step.input_type}")
                    print()
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
                print(f"\n{'=' * 60}")
                print(f"RESULT:\n{response}\n{'=' * 60}\n")

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