"""
Smoke tests for the multi-agent system.

These tests provide quick sanity checks to ensure that the
agents behave as expected. Each test defines a task,
arguments and simple conditions on the output. Running
through the entire suite should take only a few minutes and
will consume a small number of tokens.

To execute the tests, run ``python -m evals.smoke_tests``
from the repository root. Note that actual API calls will
be made and you will be charged for tokens when using
OpenAI. You can set ``USE_OLLAMA=true`` in your environment
to run the tests against a local model instead.
"""
from __future__ import annotations

import sys
import importlib

from agents import AgentManager


# Define a minimal set of test cases. Each case describes a
# task name, input arguments and simple assertions about the
# output. The assertions use substring checks to avoid
# brittle comparisons.
TESTS = [
    {
        "task": "Summarize Medical Text",
        "args": {
            "text": "Diabetes mellitus type 2 is a chronic metabolic disorder characterised by insulin resistance and hyperglycaemia.",
        },
        "must_contain": ["insulin", "glucose"],
    },
    {
        "task": "Write and Refine Research Article",
        "args": {
            "topic": "Artificial Intelligence in Radiology",
            "outline": "Introduction, Applications, Limitations, Future",
        },
        "must_contain": ["radiology", "applications", "limitations"],
    },
    {
        "task": "Sanitize Medical Data (PHI)",
        "args": {
            "medical_data": "Patient John Miller, born 12/03/1980, diagnosed with hypertension.",
        },
        "must_not_contain": ["John", "12/03/1980"],
    },
]


def run_tests() -> None:
    """Execute all smoke tests and print results to stdout."""
    manager = AgentManager(max_retries=1, verbose=False)
    passed = 0
    for idx, case in enumerate(TESTS):
        task = case["task"]
        args = case["args"]
        print(f"Running test {idx + 1}/{len(TESTS)}: {task}")
        try:
            if task == "Summarize Medical Text":
                agent = manager.get_agent("summarize")
                result = agent.execute(args["text"])
            elif task == "Write and Refine Research Article":
                agent_write = manager.get_agent("write_article")
                agent_refine = manager.get_agent("refiner")
                draft = agent_write.execute(args["topic"], args.get("outline"))
                result = agent_refine.execute(draft)
            elif task == "Sanitize Medical Data (PHI)":
                agent = manager.get_agent("sanitize_data")
                result = agent.execute(args["medical_data"])
            else:
                print(f"  Unknown task: {task}")
                continue
            # Assertions
            ok = True
            for must in case.get("must_contain", []):
                if must.lower() not in result.lower():
                    print(f"  ❌ Expected '{must}' to appear in result")
                    ok = False
            for must_not in case.get("must_not_contain", []):
                if must_not.lower() in result.lower():
                    print(f"  ❌ Expected '{must_not}' to be removed from result")
                    ok = False
            if ok:
                passed += 1
                print("  ✅ Passed")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    print(f"\n{passed}/{len(TESTS)} tests passed.")


if __name__ == "__main__":
    run_tests()