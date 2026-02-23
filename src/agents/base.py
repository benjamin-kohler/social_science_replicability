"""Base agent class for the replication system."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from ..models.config import Config, get_chat_model
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the replication system.

    This provides a common interface for interacting with LLMs
    via LangChain chat models, supporting any provider.
    """

    def __init__(
        self,
        config: Config,
        name: str,
        role: str,
        goal: str,
        chat_model: Optional[BaseChatModel] = None,
    ):
        """Initialize the agent.

        Args:
            config: Configuration object.
            name: Agent name.
            role: Agent's role description.
            goal: Agent's goal/objective.
            chat_model: Optional LangChain chat model for dependency injection.
                If None, creates one from config via get_chat_model().
        """
        self.config = config
        self.name = name
        self.role = role
        self.goal = goal
        self._chat_model = chat_model

        logger.info(
            f"Initialized {self.name} agent with "
            f"{config.langgraph.default_provider}/{config.langgraph.default_model}"
        )

    @property
    def chat_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model."""
        if self._chat_model is None:
            self._chat_model = get_chat_model(self.config)
        return self._chat_model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature (unused with LangChain, kept for API compat).
            max_tokens: Override default max tokens (unused with LangChain, kept for API compat).
            response_format: Expected response format (unused with LangChain, kept for API compat).

        Returns:
            Generated response text.
        """
        if system_prompt is None:
            system_prompt = f"You are {self.name}, a {self.role}. Your goal is: {self.goal}"

        logger.debug(f"{self.name} generating response...")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        response = self.chat_model.invoke(messages)
        content = response.content
        # LangChain may return content as a list of blocks (e.g., newer OpenAI models
        # with reasoning). Extract only text blocks, skip reasoning/other blocks.
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    # Only extract from text-type blocks, skip reasoning blocks
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif "text" in block and block.get("type") != "reasoning":
                        parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
                elif hasattr(block, "text") and getattr(block, "type", None) != "reasoning":
                    parts.append(block.text)
            content = "\n".join(parts)
        return content

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a JSON response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            Parsed JSON response.
        """
        # Append JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."

        response = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
        )

        # Parse JSON from response
        try:
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove ```json or ``` prefix and trailing ```
                cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned)
                cleaned = cleaned.strip()

            if cleaned.startswith("{"):
                return json.loads(cleaned)
            else:
                # Try to find JSON in response
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run the agent's main task.

        This method should be implemented by each specific agent.
        """
        pass
