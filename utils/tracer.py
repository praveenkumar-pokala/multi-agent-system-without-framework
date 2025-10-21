"""
Utilities for tracing agent interactions.

The ``Tracer`` class encapsulates the recording of
interaction history between agents, tools and the user. It
accumulates an ``APExchange`` and persists it to disk as a
JSON Lines file when ``finalize`` is called. Each task
receives its own exchange identified by ``task_id``.

Example:

    from agents.protocol import APMessage
    from utils.tracer import Tracer

    tracer = Tracer("task-123")
    tracer.log(APMessage(role="user", sender="user", content="Hello"))
    ...
    tracer.finalize(verdict="pass", prompt_tokens=10, output_tokens=5)

This will write a JSON representation of the exchange to
``traces/{task_id}.jsonl``. Multiple calls to ``finalize``
append entries to the same file, allowing the UI to display
history over time.
"""
import json
import os
import time
from typing import Optional

from agents.protocol import APExchange, APMessage
from utils.logger import logger

# Directory used to store trace files. A caller may set
# TRACE_DIR via the environment to override the default.
TRACE_DIR = os.environ.get("TRACE_DIR", "traces")
os.makedirs(TRACE_DIR, exist_ok=True)


class Tracer:
    """Recorder for a single task exchange."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._start = time.time()
        self.exchange = APExchange(task_id=task_id, messages=[])

    def log(self, msg: APMessage) -> None:
        """Append a message to the current exchange."""
        self.exchange.messages.append(msg)

    def finalize(
        self,
        verdict: Optional[str] = None,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Finalize the current exchange and persist it.

        Records latency and token usage, assigns the optional
        verdict and writes the exchange as a single line of
        JSON into ``{TRACE_DIR}/{task_id}.jsonl``. If the file
        exists the entry is appended.
        """
        self.exchange.latency_ms = int((time.time() - self._start) * 1000)
        self.exchange.cost_tokens_prompt += prompt_tokens
        self.exchange.cost_tokens_output += output_tokens
        if verdict is not None:
            self.exchange.verdict = verdict
        path = os.path.join(TRACE_DIR, f"{self.task_id}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.exchange.dict()) + "\n")
            logger.info(f"Trace saved to {path}")
        except Exception as e:
            logger.error(f"Failed to write trace file {path}: {e}")