"""
Agent that removes Protected Health Information (PHI) from medical data.

This tool takes raw medical data, instructs the LLM to
remove any PHI and returns the sanitized result.
"""
from .agent_base import AgentBase


class SanitizeDataTool(AgentBase):
    def __init__(self, max_retries: int = 3, verbose: bool = True) -> None:
        super().__init__(name="SanitizeDataTool", max_retries=max_retries, verbose=verbose)

    def execute(self, medical_data: str) -> str:
        """Sanitize the given medical data by removing PHI."""
        messages = [
            {"role": "system", "content": "You are an AI assistant that sanitizes medical data by removing Protected Health Information (PHI)."},
            {
                "role": "user",
                "content": (
                    "Remove all PHI from the following data:\n\n"
                    f"{medical_data}\n\nSanitized Data:"
                ),
            },
        ]
        sanitized_data = self.call_openai(messages, max_tokens=500)
        return sanitized_data