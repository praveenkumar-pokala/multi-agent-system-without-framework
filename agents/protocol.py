"""
Protocol definitions for agent messages and exchanges.

This module defines data models using Pydantic for
representing interactions between agents, users and tools.
The models are kept lightweight so that they can be
serialized as JSON. Each `APMessage` records a single
utterance from an agent or user and `APExchange` bundles
multiple messages together with metadata such as token
usage and latency.

These models are intended to provide a stable contract
between components of the system (the Streamlit UI, agent
implementations and any downstream analytics). They can
also be written to disk as JSONL records via the
``Tracer`` class in ``utils/tracer.py``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field

Role = Literal["user", "agent", "validator", "system", "tool"]


class APMessage(BaseModel):
    """Represents a single message in an agent exchange.

    Each message has a unique identifier, a role indicating
    who sent the message, the sender name, the raw content
    and optional tool invocation details. A timestamp is
    automatically assigned when the object is created.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    sender: str
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class APExchange(BaseModel):
    """A collection of messages belonging to a single task.

    In addition to the list of messages, the exchange holds
    metadata about cost, latency and the final verdict of the
    validation process. A task identifier ties together all
    communications related to a single user request.
    """

    task_id: str
    messages: List[APMessage] = []
    cost_tokens_prompt: int = 0
    cost_tokens_output: int = 0
    latency_ms: Optional[int] = None
    verdict: Optional[str] = None