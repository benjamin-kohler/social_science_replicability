"""Base agent class for the replication system."""

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import OpenAI
from anthropic import Anthropic

from ..models.config import Config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the replication system.

    This provides a common interface for interacting with LLMs,
    regardless of the underlying provider (OpenAI, Anthropic, etc.).
    """

    def __init__(
        self,
        config: Config,
        name: str,
        role: str,
        goal: str,
    ):
        """Initialize the agent.

        Args:
            config: Configuration object.
            name: Agent name.
            role: Agent's role description.
            goal: Agent's goal/objective.
        """
        self.config = config
        self.name = name
        self.role = role
        self.goal = goal
        self.provider = config.open_agent.default_provider
        self.model = config.open_agent.default_model
        self._client = None

        logger.info(f"Initialized {self.name} agent with {self.provider}/{self.model}")

    @property
    def client(self):
        """Get or create the LLM client."""
        if self._client is None:
            if self.provider == "openai":
                self._client = OpenAI(api_key=self.config.openai_api_key)
            elif self.provider == "anthropic":
                self._client = Anthropic(api_key=self.config.anthropic_api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

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
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            response_format: Expected response format ('json' for JSON mode).

        Returns:
            Generated response text.
        """
        temp = temperature or self.config.open_agent.temperature
        tokens = max_tokens or self.config.open_agent.max_tokens

        if system_prompt is None:
            system_prompt = f"You are {self.name}, a {self.role}. Your goal is: {self.goal}"

        logger.debug(f"{self.name} generating response...")

        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, temp, tokens, response_format)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, temp, tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[str] = None,
    ) -> str:
        """Generate response using OpenAI."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using Anthropic."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

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
            response_format="json" if self.provider == "openai" else None,
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from response
            if response.strip().startswith("{"):
                return json.loads(response)
            else:
                # Try to find JSON in response
                import re
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
