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
}

INSTRUCTION_KEYS = tuple(INSTRUCTIONS)


def format_instructed_query(instruction: str, problem: str) -> str:
    return f"Instruct: {instruction}\n\nQuery: {problem}"
