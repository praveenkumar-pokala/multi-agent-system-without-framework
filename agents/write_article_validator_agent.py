"""
Validator for research articles.

This agent assesses whether a generated research article
covers the topic comprehensively, follows a logical
structure and maintains academic standards. It returns a
brief analysis and a rating on a 1–5 scale.
"""
from .agent_base import AgentBase


class WriteArticleValidatorAgent(AgentBase):
    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        super().__init__(name="WriteArticleValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic: str, article: str) -> str:
        """Validate an article against the given topic."""
        system_message = "You are an AI assistant that validates research articles."
        user_content = (
            "Given the topic and the article, assess whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
            "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Topic: {topic}\n\n"
            f"Article:\n{article}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
        validation = self.call_openai(messages, max_tokens=512)
        return validation