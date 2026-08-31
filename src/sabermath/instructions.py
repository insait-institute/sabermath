INSTRUCTIONS: dict[str, str | None] = {
    "p0": None,
    "p1": (
        "You are an expert mathematical retrieval engine. Your job is to "
        "retrieve sample problems that are *conceptually close* to a target "
        "problem, judged by the *mathematical ideas and solution techniques "
        "actually used* (not surface wording)."
    ),
    "p2": (
        "Given a mathematical query, retrieve documents that share identical "
        "problem-solving structure and logical steps, even if they use "
        "entirely different variables, symbols, or natural language phrasing."
    ),
    "p3": "Find the most relevant document based on the query.",
    "pm": (
        "Given a math problem query, retrieve documents that are "
        "mathematically relevant to the query"
    ),
}

INSTRUCTION_KEYS = tuple(INSTRUCTIONS)

# The vendors' own wrapper, single newline. The double-newline variant is kept
# as a name only, so a stored result that records it stays readable.
INSTRUCTION_TEMPLATES: dict[str, str] = {
    "canonical": "Instruct: {instruction}\nQuery: {query}",
}

DEFAULT_INSTRUCTION_TEMPLATE = "canonical"


def format_instructed_query(
    instruction: str,
    problem: str,
    template: str = DEFAULT_INSTRUCTION_TEMPLATE,
) -> str:
    try:
        pattern = INSTRUCTION_TEMPLATES[template]
    except KeyError:
        raise ValueError(
            f"Unknown instruction template '{template}' - valid: "
            f"{', '.join(INSTRUCTION_TEMPLATES)}"
        ) from None
    return pattern.format(instruction=instruction, query=problem)
