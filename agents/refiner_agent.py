"""
Agent that refines research article drafts.

The refiner improves language, coherence and academic
quality of a draft article. It inherits from ``AgentBase``
and sends a multipart message using the OpenAI function
calling format when available. It returns the refined
article as plain text.
"""
from .agent_base import AgentBase


class RefinerAgent(AgentBase):
    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        super().__init__(name="RefinerAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, draft: str) -> str:
        """Refine a research article draft."""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert editor who refines and enhances research articles for clarity, coherence, and academic quality.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please refine the following research article draft to improve its language, coherence, and overall quality:\n\n"
                            f"{draft}\n\nRefined Article:"
                        ),
                    }
                ],
            },
        ]
        refined_article = self.call_openai(messages, temperature=0.5, max_tokens=2048)
        return refined_article