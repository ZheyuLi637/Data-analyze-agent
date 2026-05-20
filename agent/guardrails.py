from __future__ import annotations


UNSAFE_TERMS = (
    "delete local files",
    "delete files",
    "remove files",
    "inspect my computer",
    "shell command",
    "shell commands",
    "bash",
    "terminal",
    "arbitrary python",
    "custom python",
    "execute code",
    "run python",
    "run_python",
    "bypass validation",
    "ignore your allowed tools",
    "ignore allowed tools",
)


def evaluate_guardrail(goal: str) -> dict:
    normalized = goal.lower()
    matched = [term for term in UNSAFE_TERMS if term in normalized]
    if not matched:
        return {"blocked": False, "reason": "", "matched_terms": []}
    return {
        "blocked": True,
        "reason": "The request asks the agent to bypass safe tool boundaries or execute arbitrary local actions.",
        "matched_terms": matched,
    }


def guardrail_message(guardrail: dict) -> str:
    return (
        "Safety guardrail blocked this request. This prototype can analyze uploaded CSV data only through "
        "the predefined safe tools. It will not execute arbitrary Python, shell commands, inspect local files, "
        "or bypass tool validation."
    )

