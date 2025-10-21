"""Agent manager and factory.

This module exposes all available agents and provides an
``AgentManager`` for retrieving them by name. It acts as a
simple factory to instantiate tools and validator agents
with consistent retry and verbosity settings. New agents
should be added here to make them available to the rest
of the application.
"""
from .summarize_tool import SummarizeTool
from .write_article_tool import WriteArticleTool
from .sanitize_data_tool import SanitizeDataTool
from .summarize_validator_agent import SummarizeValidatorAgent
from .write_article_validator_agent import WriteArticleValidatorAgent
from .sanitize_data_validator_agent import SanitizeDataValidatorAgent
from .refiner_agent import RefinerAgent
from .validator_agent import ValidatorAgent


class AgentManager:
    """Simple registry for instantiating agents by name."""

    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        self.agents = {
            "summarize": SummarizeTool(max_retries=max_retries, verbose=verbose),
            "write_article": WriteArticleTool(max_retries=max_retries, verbose=verbose),
            "sanitize_data": SanitizeDataTool(max_retries=max_retries, verbose=verbose),
            "summarize_validator": SummarizeValidatorAgent(max_retries=max_retries, verbose=verbose),
            "write_article_validator": WriteArticleValidatorAgent(max_retries=max_retries, verbose=verbose),
            "sanitize_data_validator": SanitizeDataValidatorAgent(max_retries=max_retries, verbose=verbose),
            "refiner": RefinerAgent(max_retries=max_retries, verbose=verbose),
            "validator": ValidatorAgent(max_retries=max_retries, verbose=verbose),
        }

    def get_agent(self, agent_name: str):
        """Retrieve an agent by name from the registry.

        Raises a ValueError if the agent does not exist.
        """
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent '{agent_name}' not found.")
        return agent