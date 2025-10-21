"""
Agent that summarizes medical texts.

This tool takes a block of medical text and produces a
concise summary. It inherits from ``AgentBase`` and uses
the ``call_openai`` method to communicate with the LLM.
"""
from .agent_base import AgentBase


class SummarizeTool(AgentBase):
    def __init__(self, max_retries: int = 3, verbose: bool = True) -> None:
        super().__init__(name="SummarizeTool", max_retries=max_retries, verbose=verbose)

    def execute(self, text: str) -> str:
        """Summarize the given medical text."""
        messages = [
            {"role": "system", "content": "You are an AI assistant that summarizes medical texts."},
            {
                "role": "user",
                "content": (
                    "Please provide a concise summary of the following medical text:\n\n"
                    f"{text}\n\nSummary:"
                ),
            },
        ]
        summary = self.call_openai(messages, max_tokens=300)
        return summary