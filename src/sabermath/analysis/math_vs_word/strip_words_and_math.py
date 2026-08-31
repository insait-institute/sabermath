from datasets import load_dataset
import re
import argparse
import yaml

from embed import get_top5_candidates

parser = argparse.ArgumentParser()
parser.add_argument("--config_file")
args = parser.parse_args()

config_path = args.config_file

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

targets_fixed_latex_dataset = config["hf_datasets"]["targets_fixed_latex"]
candidates_fixed_latex_dataset = config["hf_datasets"]["candidates_fixed_latex"]

targets_filtered_fixed = config["hf_datasets"]["targets_maths_words_fixed"]
candidates_filtered_fixed = config["hf_datasets"]["candidates_maths_words_fixed"]

fixed_targets = load_dataset(targets_fixed_latex_dataset)["train"]
fixed_candidates = load_dataset(candidates_fixed_latex_dataset)["train"]

math_pattern = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)


def process(example):

    content = example["problem_fixed"]

    if content is None:
        return {
            "problem_math_expr": None,
            "problem_text_only": None,
            "solution_math_expr": None,
            "solution_text_only": None,
        }

    math_expressions = math_pattern.findall(content)
    math_expressions = [expr.strip() for expr in math_expressions]
    joined = ", ".join(math_expressions)
    text_only = math_pattern.sub("", content).strip()

    sol_content = example["solution_fixed"]

    sol_math_expressions = math_pattern.findall(sol_content)
    sol_math_expressions = [expr.strip() for expr in sol_math_expressions]
    sol_joined = ", ".join(sol_math_expressions)
    sol_text_only = math_pattern.sub("", sol_content).strip()

    return {
        "problem_math_expr": joined,
        "problem_text_only": text_only,
        "solution_math_expr": sol_joined,
        "solution_text_only": sol_text_only,
    }


fixed_targets = fixed_targets.map(process)
fixed_candidates = fixed_candidates.map(process)

good_targets = fixed_targets.filter(
    lambda x: len(x["problem_math_expr"]) != 0
    and len(x["problem_text_only"]) != 0
    and all(
        len(y["problem_math_expr"]) + len(y["solution_math_expr"]) != 0
        and len(y["problem_text_only"]) + len(y["solution_text_only"]) != 0
        for y in [fixed_candidates[s] for s in get_top5_candidates(x)]
    )
)

good_targets.push_to_hub(targets_filtered_fixed)
fixed_candidates.push_to_hub(candidates_filtered_fixed)
