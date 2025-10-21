"""
Model abstraction with pluggable providers.

This module wraps calls to large language models (LLMs) and
allows switching between the OpenAI API and a locally
hosted model served via Ollama. Selection is controlled
through environment variables so that the same code can run
in production (with OpenAI) or offline (with Ollama).

If ``USE_OLLAMA=true`` is present in the environment the
``llm_call`` function will send a request to an Ollama
instance at ``http://localhost:11434/api/chat``. Otherwise
the OpenAI Python client is used. When using OpenAI, model
name can be set via ``OPENAI_MODEL``; for Ollama it is
``OLLAMA_MODEL``. Token usage information is returned
consistently in both cases.
"""
import os
from typing import List, Dict, Tuple, Any

USE_OLLAMA = os.environ.get("USE_OLLAMA", "false").lower() == "true"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")


def llm_call(messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
    """Invoke the configured language model with chat messages.

    Args:
        messages: A list of dictionaries following the OpenAI
            chat format, each containing ``role`` and
            ``content`` keys (and optionally others).

    Returns:
        A tuple ``(content, usage)`` where ``content`` is the
        assistant's reply as plain text and ``usage`` is a
        dictionary with ``prompt_tokens`` and ``output_tokens``.

    Raises:
        Exception: If the underlying provider call fails.
    """
    if USE_OLLAMA:
        # Perform a local request to the Ollama API. We keep this
        # light and avoid adding a dependency on ``requests`` so
        # that environments without internet can still run the
        # system. Instead we use the built-in urllib library.
        import json
        import urllib.request
        import urllib.error

        url = "http://localhost:11434/api/chat"
        payload = json.dumps({"model": OLLAMA_MODEL, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise Exception(f"Failed to call Ollama model at {url}: {e}")
        # Ollama returns a nested structure similar to OpenAI. Extract
        # content and ignore token counts (not provided by Ollama).
        content = data.get("message", {}).get("content", "")
        return content, {"prompt_tokens": 0, "output_tokens": 0}
    else:
        # Use OpenAI's API. Delay import of openai so that the
        # dependency is not required when running in Ollama mode.
        from openai import OpenAI

        client = OpenAI()
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
            )
        except Exception as e:
            raise Exception(f"OpenAI chat completion failed: {e}")
        content = resp.choices[0].message.content
        usage = resp.usage or {}
        return content, {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
        }