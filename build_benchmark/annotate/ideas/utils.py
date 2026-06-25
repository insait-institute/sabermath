import re


def get_solution_label(index: int) -> str:
    return f"model_solution_{index}"


def extract_single_answer(solution: str) -> str:
    matches = re.findall(r"\\boxed\{(.*?)\}", solution)
    if matches:
        return matches[-1]
    return ""


def extract_answers(solutions: list[str]) -> list[str]:
    return [extract_single_answer(solution) for solution in solutions]
