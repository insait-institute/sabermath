# Idea - Guide
This directory holds holds the code for a Extracting the core ideas of problems in the Databank. Extraction is done via an OpenAI model (could either be via the Official OpenAI API or via `vllm serve`-based API).

## General Syntax
The `extract_ideas` tool takes a dataset from HuggingFace and annotates extract a single core idea (20-30 words) from each problem + solution. The syntax is:
```
python extract_tags.py HF_PATH (--datasets DATASETS) (--statement-column COL) (--solution-column COL) (--out OUT_PATH) (--out-column OUT_COLUMN) (--model MODEL) (--api-url URL) (--silent)
```

Here the arguments are:

- `HF_PATH` - HuggingFace Path where the dataset is located
- `--datasets DATASETS` - Comma-separated list of the datasets in the path to annotate (Default: all)
- `--statement-column COL` - Name of the column containing the problem statement. It must be present in each of the datasets (Default: `problem`)
- `--solution-column COL` - Name of the column containing the problem solution. It must be present in each of the datasets. (Default: `solution`)
- `--out OUT_PATH` - HuggingFace Path where to save the resulted outputs. (Default: `{HF_PATH}_ideas`)
- `--out-column OUT_COLUMN` - Name of the column where to save the ideas (Default: `idea`)
- `--model` - OpenAI model to use (Default: `<empty>`)
- `--api-url URL` - OpenAI API URL to use (Default: `https://api.openai.com/v1`)
- `--silent` - Block info printing

## Running via OpenAI Official API
To run the idea extraction script using the official OpenAI API leave `--api-url` to its default value. You must specify a model to use (e.g. `gpt-5-mini`). The minimal command becomes:
```
python extract_ideas.py <hf_path> --model <model>
```

## Running via vLLM
You may want to use `gpt-oss-20b` of `gpt-oss-120b`. You may do this locally via `vllm serve`. For example, run `gpt-oss-20b` on port `8086`:

```
vllm server openai/gpt-oss-20b --async-scheduling --host 0.0.0.0 --port 8086
```

Then leave `--model` to its default empty value but set the `--api-url` to point to the vLLM API. Here is a minimal example:

```
python extract_ideas.py <hf_path> --api-url http://localhost:8086/v1
```

**NOTICE**: As of version `0.11.0` vLLM requires GPU UIDs to be integers starting with 0. In many cases proper GPU IDs are not numeric. To handle this please call:
```
export CUDA_VISIBLE_DEVICES=0,1,...,(NUM_OF_GPUs - 1)
```
