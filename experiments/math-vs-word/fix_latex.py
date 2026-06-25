from datasets import load_dataset, Dataset
import sys
from pathlib import Path
from loguru import logger
from huggingface_hub import login
import os
import argparse
import tqdm
import yaml

login(token=os.environ["HF_TOKEN"])

sys.path.append(str(Path(__file__).resolve().parents[2]))
from llmteach.executor import QueryExecutor
from llmteach.api import APIQuery
from llmteach.postprocess import fix_thinking


SAVE_EVERY_SUCCESSFUL_PROMPTS = 1000


def extract_boxed(text: str) -> list[str]:
    results = []
    i = 0
    while i < len(text):
        if text.startswith(r"\boxed{", i):
            i += len(r"\boxed{")
            brace_level = 1
            start = i

            while i < len(text) and brace_level > 0:
                if text[i] == "{":
                    brace_level += 1
                elif text[i] == "}":
                    brace_level -= 1
                i += 1

            # Extract content without outer braces
            results.append(text[start : i - 1])
        else:
            i += 1

    return results


PROMPT = r"""Role: You are a LaTeX Formatting Expert specializing in mathematical notation standardization.

Task: You will be provided with mathematical text (problems or solutions). Your task is to standardize the LaTeX formatting and enclose the entire final result within a \boxed{{...}} command.

Strict Constraints:
1. DELIMITER CONVERSION: Replace all instances of \( ... \) and \\( ... \\) with standard dollar signs.
   - Use $...$ for inline text.
   - Use $$...$$ for display equations.
2. UNIVERSAL MATH TAGGING: Apply math mode ($...$) to every single mathematical element without exception.
3. CONTENT INTEGRITY: Do not solve the problem or edit the prose.
4. FINAL WRAPPING: The entire output must be contained within \boxed{{ <your_formatted_text_here> }}.
5. NO VERBOSITY: Provide ONLY the \boxed{{...}} block.

Example Transformation:
Input: If the radius r is 5, find the area. Use \( \pi \).
Output: \boxed{{If the radius $r$ is $5$, find the area. Use $\pi$.}}
"""


def main():

    print("Begin:")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Path to a config file")

    args = parser.parse_args()
    config_path = args.config_file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    targets_dataset = config["hf_datasets"]["original_targets"]
    candidates_dataset = config["hf_datasets"]["original_candidates"]
    fixed_targets_dataset = config["hf_datasets"]["targets_fixed_latex"]
    fixed_candidates_dataset = config["hf_datasets"]["candidates_fixed_latex"]

    targets = load_dataset(targets_dataset)["train"]
    candidates = load_dataset(candidates_dataset)["train"]

    # Load already-fixed datasets
    fixed_targets = load_dataset(fixed_targets_dataset)["train"]
    fixed_candidates = load_dataset(fixed_candidates_dataset)["train"]

    querier = APIQuery(
        model="openai/gpt-oss-120b",
        api="vllm",
        max_tokens=50000,
        read_cost=0.15,
        write_cost=0.60,
        concurrent_requests=16,
        reasoning_effort="high",
        human_readable_id="GPT OSS 120B (high-2)",
        date="2025-08-05",
    )

    executor = QueryExecutor(
        querier=querier,
        prompt_template=PROMPT,
        system_prompt="You are a helpful expert.",  # Optional
    )

    target_dict = {target["id"]: dict(target) for target in targets}
    candidates_dict = {candidate["id"]: dict(candidate) for candidate in candidates}

    # Merge existing fixed fields into the original dicts
    for fixed_target in fixed_targets:
        prob_id = fixed_target["id"]
        if prob_id not in target_dict:
            logger.warning(
                f"Fixed target id {prob_id} not present in original targets; skipping."
            )
            continue

        if "problem_fixed" in fixed_target:
            target_dict[prob_id]["problem_fixed"] = fixed_target["problem_fixed"]
        if "solution_fixed" in fixed_target:
            target_dict[prob_id]["solution_fixed"] = fixed_target["solution_fixed"]

    for fixed_candidate in fixed_candidates:
        prob_id = fixed_candidate["id"]
        if prob_id not in candidates_dict:
            logger.warning(
                f"Fixed candidate id {prob_id} not present in original candidates; skipping."
            )
            continue

        if "problem_fixed" in fixed_candidate:
            candidates_dict[prob_id]["problem_fixed"] = fixed_candidate["problem_fixed"]
        if "solution_fixed" in fixed_candidate:
            candidates_dict[prob_id]["solution_fixed"] = fixed_candidate[
                "solution_fixed"
            ]

    dirty_since_last_save = {
        "target": False,
        "candidate": False,
    }

    def build_target_dataset() -> Dataset:
        target_data = [dict(row) for row in target_dict.values()]
        for row in target_data:
            row.setdefault("problem_fixed", None)
            row.setdefault("solution_fixed", None)

        return Dataset.from_list(target_data)

    def build_candidate_dataset() -> Dataset:
        candidate_data = [dict(row) for row in candidates_dict.values()]
        for row in candidate_data:
            row.setdefault("problem_fixed", None)
            row.setdefault("solution_fixed", None)

        return Dataset.from_list(candidate_data)

    def push_current_results_to_hub(
        reason: str,
        force: bool = False,
        raise_on_error: bool = False,
    ) -> bool:
        """
        Save current results to Hugging Face.

        During runtime, this is called only after every 1000 successful prompts.
        It pushes only datasets that changed since the previous successful save.
        At final save, force=True pushes both datasets.
        """
        all_ok = True

        if force or dirty_since_last_save["target"]:
            try:
                logger.info(
                    f"Saving {reason} target results to {fixed_targets_dataset}..."
                )
                hf_target_dataset = build_target_dataset()
                hf_target_dataset.push_to_hub(fixed_targets_dataset)
                dirty_since_last_save["target"] = False
                logger.info(
                    f"Successfully saved target results to {fixed_targets_dataset}."
                )
            except Exception as save_error:
                all_ok = False
                logger.exception(
                    f"Saving target results to Hugging Face failed: {save_error}"
                )
                if raise_on_error:
                    raise

        if force or dirty_since_last_save["candidate"]:
            try:
                logger.info(
                    f"Saving {reason} candidate results to {fixed_candidates_dataset}..."
                )
                hf_candidates_dataset = build_candidate_dataset()
                hf_candidates_dataset.push_to_hub(fixed_candidates_dataset)
                dirty_since_last_save["candidate"] = False
                logger.info(
                    f"Successfully saved candidate results to {fixed_candidates_dataset}."
                )
            except Exception as save_error:
                all_ok = False
                logger.exception(
                    f"Saving candidate results to Hugging Face failed: {save_error}"
                )
                if raise_on_error:
                    raise

        return all_ok

    candidate_indices = list(
        set(candidate for target in targets for candidate in target["candidates"])
    )

    targets_problem_prompts = [
        {
            "id": target["id"],
            "maths": target["problem"],
            "type": "target",
            "field": "problem",
        }
        for target in targets
        if target_dict[target["id"]].get("problem_fixed") is None
    ]

    candidates_problem_prompts = [
        {
            "id": candidates[i]["id"],
            "maths": candidates[i]["problem"],
            "type": "candidate",
            "field": "problem",
        }
        for i in candidate_indices
        if candidates_dict[candidates[i]["id"]].get("problem_fixed") is None
    ]

    targets_sol_prompts = [
        {
            "id": target["id"],
            "maths": target["solution"],
            "type": "target",
            "field": "solution",
        }
        for target in targets
        if target_dict[target["id"]].get("solution_fixed") is None
    ]

    candidates_sol_prompts = [
        {
            "id": candidates[i]["id"],
            "maths": candidates[i]["solution"],
            "type": "candidate",
            "field": "solution",
        }
        for i in candidate_indices
        if candidates_dict[candidates[i]["id"]].get("solution_fixed") is None
    ]

    problem_prompts = targets_problem_prompts + candidates_problem_prompts
    sol_prompts = targets_sol_prompts + candidates_sol_prompts

    all_prompts = problem_prompts + sol_prompts

    print("Fixed LaTeX fields still missing / prompts to run:")
    print(
        f"  Target problem prompts:    {len(targets_problem_prompts)} / {len(targets)}"
    )
    print(f"  Target solution prompts:   {len(targets_sol_prompts)} / {len(targets)}")
    print(
        f"  Candidate problem prompts: {len(candidates_problem_prompts)} / {len(candidate_indices)} referenced candidates"
    )
    print(
        f"  Candidate solution prompts:{len(candidates_sol_prompts)} / {len(candidate_indices)} referenced candidates"
    )
    print(f"  Total problem prompts:     {len(problem_prompts)}")
    print(f"  Total solution prompts:    {len(sol_prompts)}")
    print(f"  Total prompts:             {len(all_prompts)}")

    logger.info(f"Running {len(problem_prompts)} problems and {len(sol_prompts)}")

    todo_prompts = all_prompts
    successful_prompt_count = 0
    unsaved_successful_prompt_count = 0

    while len(todo_prompts) != 0:

        failed_prompts = []

        for chunk_start in range(0, len(todo_prompts), SAVE_EVERY_SUCCESSFUL_PROMPTS):
            chunk_prompts = todo_prompts[
                chunk_start : chunk_start + SAVE_EVERY_SUCCESSFUL_PROMPTS
            ]

            logger.info(
                f"Processing prompt chunk "
                f"{chunk_start} to {chunk_start + len(chunk_prompts)} "
                f"out of {len(todo_prompts)} current todo prompts."
            )

            for idx, messages, detailed_cost in tqdm.tqdm(
                executor.execute(chunk_prompts),
                total=len(chunk_prompts),
            ):

                prompt = chunk_prompts[idx]
                prob_id = prompt["id"]
                prompt_type = prompt["type"]
                field = prompt["field"]

                last_content = messages[-1]["content"]

                if isinstance(last_content, list):
                    last_content = last_content[-1]["content"]

                final_answer = fix_thinking(last_content)
                maths = extract_boxed(final_answer)

                if len(maths) != 1:
                    logger.warning(
                        f"Parsing failed for {prompt_type} {prob_id}. "
                        f"Queuing for retry."
                    )
                    failed_prompts.append(prompt)
                else:
                    if prompt_type == "target":
                        target_dict[prob_id][f"{field}_fixed"] = maths[0]
                        dirty_since_last_save["target"] = True
                    elif prompt_type == "candidate":
                        candidates_dict[prob_id][f"{field}_fixed"] = maths[0]
                        dirty_since_last_save["candidate"] = True
                    else:
                        raise ValueError(f"Unknown type {prompt_type}")

                    successful_prompt_count += 1
                    unsaved_successful_prompt_count += 1

                    if unsaved_successful_prompt_count >= SAVE_EVERY_SUCCESSFUL_PROMPTS:
                        logger.info(
                            f"Reached {successful_prompt_count} total successful prompts. "
                            f"Saving partial results to Hugging Face..."
                        )

                        saved_ok = push_current_results_to_hub(
                            reason=f"partial after {successful_prompt_count} successful prompts",
                            force=False,
                            raise_on_error=False,
                        )

                        if saved_ok:
                            unsaved_successful_prompt_count = 0

        logger.warning(f"Redoing {len(failed_prompts)}")

        todo_prompts = failed_prompts

    print("Final save to Hugging Face datasets...")
    push_current_results_to_hub(
        reason="final",
        force=True,
        raise_on_error=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
