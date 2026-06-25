import asyncio
import argparse
import json, os
from typing import Sequence

import httpx
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from prompts import extract_core_idea_query


async def call_openai_json(
    message: str,
    model: str,
    client: AsyncOpenAI,
    reasoning: str = "high",
    retries: int = 4,
) -> list | dict:
    exception = None
    while retries:
        try:
            response = await client.responses.create(
                model=model,
                input=[{"role": "user", "content": message}],
                reasoning={"effort": reasoning},
            )

            content = response.output_text
            return json.loads(content)

        except KeyboardInterrupt:
            break

        except Exception as e:
            exception = e
            retries -= 1

    raise exception


async def generate_ideas(
    statements: Sequence[str],
    solutions: Sequence[str],
    *,
    model: str = "",
    reasoning: str = "high",
    api_url: str = "https://api.openai.com/v1",
    max_concurrency: int = 512,
    show_progress_bar: bool = True,
) -> list[str]:
    if len(statements) != len(solutions):
        raise ValueError("number of statement and solutions must match")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # in case vLLM is used locally
        api_key = "sk-PLACEHOLDER"

    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )

    http_client = httpx.AsyncClient(limits=limits)
    client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key,
        http_client=http_client,
    )

    sem = asyncio.Semaphore(max_concurrency)

    async def get_core_idea(st: str, sol: str) -> str:
        async with sem:
            try:
                json_resp = await call_openai_json(
                    extract_core_idea_query(st, sol),
                    model,
                    client,
                    reasoning=reasoning,
                )
                no_core_idea = (
                    "noCoreIdea" not in json_resp
                    or json_resp["noCoreIdea"]
                    or "coreIdea" not in json_resp
                )
                if no_core_idea:
                    return ""
                else:
                    return json_resp["coreIdea"]
            except Exception as e:
                return ""

    tasks = [get_core_idea(st, sol) for st, sol in zip(statements, solutions)]
    task_desc = "Extracting Core Ideas"

    if show_progress_bar:
        core_ideas = await tqdm_asyncio.gather(*tasks, desc=task_desc)
    else:
        core_ideas = await asyncio.gather(*tasks)

    return core_ideas


async def main() -> None:
    parser = argparse.ArgumentParser(prog="generate")

    parser.add_argument("input", type=str, help="Input HuggingFace path")

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        required=False,
        help="Comma-separated list of datasets withing to path to annotate",
    )

    parser.add_argument(
        "--statement-column",
        type=str,
        default="statement",
        required=False,
        help="Name of the column with problem statements",
    )

    parser.add_argument(
        "--solution-column",
        type=str,
        default="solution",
        required=False,
        help="Name of the column with problem solutions",
    )

    parser.add_argument(
        "--out", type=str, default=None, required=False, help="Output HuggingFace path"
    )

    parser.add_argument(
        "--out-column",
        type=str,
        default="idea",
        required=False,
        help="Output HuggingFace Column",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="",
        required=False,
        help="OpenAI model to use for annotation",
    )

    parser.add_argument(
        "--reasoning",
        type=str,
        default="high",
        required=False,
        help="Reasoning effort for the model",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="https://api.openai.com/v1",
        required=False,
        help="OpenAI API URL",
    )

    parser.add_argument(
        "--silent", action="store_true", help="Prevent info logging when running"
    )

    args = parser.parse_args()
    verbose = not args.silent

    if verbose:
        print(f'[~] Loading corpus "{args.input}"...')

    try:
        ds = load_dataset(args.input)
    except Exception as e:
        print(f'[ERROR] Loading "{args.input}" failed due to: {e}')
        return

    if args.datasets is None:
        datasets = list(ds.keys())
    else:
        datasets = args.datasets.split(",")

    for name in datasets:
        if name not in ds:
            print(f'[ERROR] No dataset "{name}" in "{args.input}"')
            return
        if args.statement_column not in ds[name].column_names:
            print(
                f'[ERROR] No column "{args.statement_column}" in dataset "{args.input}:{name}"'
            )
            return
        if args.solution_column not in ds[name].column_names:
            print(
                f'[ERROR] No column "{args.solution_column}" in dataset "{args.input}:{name}"'
            )
            return

    if verbose:
        print("[+] Corpus loaded sucessfully.")

    for name in datasets:
        if verbose:
            print(f'[~] Generating ideas for dataset "{args.input}:{name}"')
        ideas = await generate_ideas(
            ds[name][args.statement_column],
            ds[name][args.solution_column],
            model=args.model,
            reasoning=args.reasoning,
            api_url=args.api_url,
            max_concurrency=512,
            show_progress_bar=verbose,
        )

        ds[name] = ds[name].add_column(args.out_column, ideas)

    out_path = f"{args.input}_ideas" if args.out is None else args.out

    if verbose:
        print("[+] Generating finished.")
        print(f'[~] Uploading to HuggingFace at "{out_path}"...')

    ds.push_to_hub(out_path)

    if verbose:
        print("[+] Finished.")


if __name__ == "__main__":
    asyncio.run(main())
