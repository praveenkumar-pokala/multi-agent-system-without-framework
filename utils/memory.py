"""
Simple memory utilities for agents.

This module provides two small classes that can be used to
store conversational context across agent invocations. A
sliding window memory retains the last ``k`` pieces of
context, while ``EntitiesMemory`` collects capitalized
tokens which often correspond to named entities. These
classes do not impose any particular serialization format
and can be composed together by concatenating their
``context()`` outputs.
"""
from collections import deque
import re
from typing import Deque, List


class SlidingMemory:
    """Maintain a fixed-size window of recent text snippets."""

    def __init__(self, k: int = 6) -> None:
        self.buf: Deque[str] = deque(maxlen=k)

    def add(self, text: str) -> None:
        """Add a new piece of text to the memory."""
        self.buf.append(text)

    def context(self) -> str:
        """Return the concatenated context from all stored snippets."""
        return "\n".join(self.buf)


class EntitiesMemory:
    """Collect naive named entities from text snippets."""

    def __init__(self) -> None:
        self.entities: dict[str, bool] = {}

    def ingest(self, text: str) -> None:
        """Extract capitalized words and add them as entities."""
        for token in re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", text):
            self.entities[token] = True

    def context(self) -> str:
        """Return a formatted string of known entities."""
        if not self.entities:
            return ""
        return "Known entities: " + ", ".join(sorted(self.entities.keys()))