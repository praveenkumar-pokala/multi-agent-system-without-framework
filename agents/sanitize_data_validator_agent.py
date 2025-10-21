"""
Validator for sanitized medical data.

This agent checks whether Protected Health Information (PHI)
has been successfully removed from the sanitized data and
rates the quality of sanitization on a 1–5 scale.
"""
from .agent_base import AgentBase


class SanitizeDataValidatorAgent(AgentBase):
    def __init__(self, max_retries: int = 2, verbose: bool = True) -> None:
        super().__init__(name="SanitizeDataValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, original_data: str, sanitized_data: str) -> str:
        """Validate that all PHI has been removed from the data."""
        system_message = (
            "You are an AI assistant that validates the sanitization of medical data by checking for the removal of Protected Health Information (PHI)."
        )
        user_content = (
            "Given the original data and the sanitized data, verify that all PHI has been removed.\n"
            "List any remaining PHI in the sanitized data and rate the sanitization process on a scale of 1 to 5, where 5 indicates complete sanitization.\n\n"
            f"Original Data:\n{original_data}\n\n"
            f"Sanitized Data:\n{sanitized_data}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]
        validation = self.call_openai(messages, max_tokens=512)
        return validation