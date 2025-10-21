"""
Reusable agent interaction patterns.

This module contains higher-level interaction patterns that
compose multiple calls to the underlying language model.
Currently it implements a simple reflection-and-revise loop
for improving outputs. The pattern accepts a task
description and a draft output, critiques the draft using a
critic and, if needed, applies a patch to produce a
revised draft. The revision process repeats up to
``max_retries`` times.

While this implementation is general-purpose, the concrete
prompts are tailored for text generation tasks. You can
adapt the critique and revision prompts for domain-specific
use cases.
"""
from typing import Optional

from utils.model import llm_call
from utils.tracer import Tracer
from agents.protocol import APMessage


# Critique prompt used by the reflection loop. The prompt
# instructs the model to identify concrete issues and
# provide a JSON patch. You can customize this to suit your
# domain.
CRITIQUE_PROMPT = (
    "You are a meticulous reviewer. Given the task description and draft, "
    "list concrete issues (if any) and propose exact fixes. Return JSON "
    "with fields: {\"issues\": [...], \"revise_required\": true/false, "
    "\"patch\": \"...\"}."
)


def reflective_improve(
    task_id: str,
    agent_name: str,
    task_desc: str,
    draft: str,
    max_retries: int = 1,
) -> str:
    """Apply a critique-and-revise loop to improve a draft.

    Args:
        task_id: A unique identifier for this interaction, used by the tracer.
        agent_name: The name of the producing agent.
        task_desc: The description of the task being performed.
        draft: The initial draft output produced by the agent.
        max_retries: Maximum number of revisions after the initial critique.

    Returns:
        The final revised draft after at most ``max_retries`` revisions.
    """
    tracer = Tracer(task_id)
    # Record the initial user and agent messages
    tracer.log(APMessage(role="user", sender="user", content=task_desc))
    tracer.log(APMessage(role="agent", sender=agent_name, content=draft))

    current_draft = draft
    for attempt in range(max_retries + 1):
        # Ask the critic to assess the draft
        critique_message = (
            f"TASK:\n{task_desc}\n\nDRAFT:\n{current_draft}\n"
        )
        critic_prompt = [
            {"role": "system", "content": CRITIQUE_PROMPT},
            {"role": "user", "content": critique_message},
        ]
        critique_response, usage1 = llm_call(critic_prompt)
        tracer.log(APMessage(role="validator", sender="critic", content=critique_response))
        # Determine whether a revision is required
        if "\"revise_required\": false" in critique_response.lower():
            tracer.finalize(verdict="pass", prompt_tokens=usage1.get("prompt_tokens", 0), output_tokens=usage1.get("output_tokens", 0))
            return current_draft
        # Ask the model to apply the patch or produce a revised draft
        revision_prompt = [
            {"role": "system", "content": "Apply the patch or produce a revised complete draft."},
            {"role": "user", "content": critique_response},
        ]
        revised, usage2 = llm_call(revision_prompt)
        tracer.log(APMessage(role="agent", sender=f"{agent_name}-reviser", content=revised))
        current_draft = revised
        if attempt == max_retries:
            tracer.finalize(
                verdict="pass_after_revise",
                prompt_tokens=usage1.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
                output_tokens=usage1.get("output_tokens", 0) + usage2.get("output_tokens", 0),
            )
            return current_draft
    # Should not reach here
    return current_draft