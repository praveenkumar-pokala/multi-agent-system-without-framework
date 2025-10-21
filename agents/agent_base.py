"""
Abstract base class for all agents.

Agents encapsulate a particular task (e.g. summarization,
writing, sanitization) and define an ``execute`` method
responsible for orchestrating calls to the language model.
This base class provides common functionality, including
retry logic, logging and an abstraction over the language
model via the ``llm_call`` function in ``utils.model``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict

from utils.logger import logger
from utils.model import llm_call


class AgentBase(ABC):
    """Base class for agents with retry and logging support."""

    def __init__(self, name: str, max_retries: int = 2, verbose: bool = True) -> None:
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent on the provided inputs."""
        raise NotImplementedError

    def call_openai(self, messages: List[Dict[str, Any]], temperature: float = 0.7, max_tokens: int = 150) -> str:
        """Call the configured language model with retry logic.

        This method wraps the lower-level ``llm_call`` function
        with retry logic and logs inputs/outputs when
        ``verbose`` is enabled. It always returns the
        assistant's reply as plain text.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to LLM:")
                    for msg in messages:
                        role = msg.get("role")
                        content_preview = str(msg.get("content"))[:120].replace("\n", " ")
                        logger.debug(f"  {role}: {content_preview}...")
                # ``llm_call`` abstracts away OpenAI vs. Ollama.
                response_text, usage = llm_call(messages)
                if self.verbose:
                    # Prepare preview string outside of the f-string to avoid backslash
                    preview = response_text[:120].replace("\n", " ")
                    logger.info(f"[{self.name}] Received response (first 120 chars): {preview}")
                return response_text
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}] Error during LLM call: {e}. Retry {retries}/{self.max_retries}")
        raise Exception(f"[{self.name}] Failed to get response after {self.max_retries} attempts")