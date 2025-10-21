"""
Agent that writes research articles on a given topic.

This tool generates a draft article based on a topic and
optional outline. It utilises the LLM via the base
class's ``call_openai`` method.
"""
from .agent_base import AgentBase


class WriteArticleTool(AgentBase):
    def __init__(self, max_retries: int = 3, verbose: bool = True) -> None:
        super().__init__(name="WriteArticleTool", max_retries=max_retries, verbose=verbose)

    def execute(self, topic: str, outline: str | None = None) -> str:
        """Generate a research article for the given topic."""
        system_message = "You are an expert academic writer."
        user_content = f"Write a research article on the following topic:\nTopic: {topic}\n\n"
        if outline:
            user_content += f"Outline:\n{outline}\n\n"
        user_content += "Article:\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
        article = self.call_openai(messages, max_tokens=1000)
        return article