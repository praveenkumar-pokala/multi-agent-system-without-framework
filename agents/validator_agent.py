"""
Generic validator agent for research articles.

This agent checks accuracy, completeness and adherence to
academic standards of research articles given a topic. It
returns a brief analysis and a rating between 1 and 5.
"""
from .agent_base import AgentBase


class ValidatorAgent(AgentBase):
    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        super().__init__(name="ValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic: str, article: str) -> str:
        """Validate a research article for completeness and quality."""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that validates research articles for accuracy, completeness, and adherence to academic standards.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Given the topic and the research article below, assess whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
                            "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
                            f"Topic: {topic}\n\n"
                            f"Article:\n{article}\n\nValidation:"
                        ),
                    }
                ],
            },
        ]
        validation = self.call_openai(messages, temperature=0.3, max_tokens=500)
        return validation