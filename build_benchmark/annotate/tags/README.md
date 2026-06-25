# Tag Maker - Guide
This directory holds the code for a Mathematical Script Tagging Script. The Tags are structured in a tree, provided via a JSON file. Tagging is done via OpenAI model (could either be via the Official OpenAI API or via `vllm serve`-based API). Multiple tags are produced for each input.

## Tree JSON Schema
The Tags must be tree-based and provided via a JSON file. Each node must follow the schema:
```
{
    "title": "<string>",
    "links": [<other node objects>]
}
```

## General Syntax
The `make_tags` tool takes a dataset from HuggingFace and annotates it with multiple tags. The result it saves in a separate dataset.

The syntax is as it follows:

```
python make_tags.py HF_PATH (--datasets DATASETS) (--columns COLUMNS) (--out OUT_PATH)
(--out-column OUT_COLUMN) (--tree TREE_PATH) (--model MODEL) (--api-url URL)
(--threshold THRESHOLD) (--silent)
```

Here the arguments are:

- `HF_PATH` - HuggingFace Path where the dataset is located
- `--datasets DATASETS` - Comma-separated list of the datasets in the path to annotate (Default: all)
- `--columns COLUMNS` - Comma-separated list of columns to feed into the Tagging Model. They must be present in each of the datasets (Default: `statement,solution`)
- `--out OUT_PATH` - HuggingFace Path where to save the resulted outputs (Default: `{HF_PATH}_tagged`)
- `--out-column OUT_COLUMN` - Name of the column where output tag lists are saved (Default: `tags`)
- `--tree TREE_PATH` - Local path where the tree's `.json` file is saved (Default: `../../data/tree.json`)
- `--model MODEL` - OpenAI model to use (Default: `<empty>`)
- `--api-url URL` - OpenAI API URL to use (Default: `https://api.openai.com/v1`)
- `--threshold THRESHOLD` - Relevance score minimum to include - check Mechanics section. Less means more liberal tag choosing and more running time (Default: `0.85`)
- `--silent` - Block info printing

## Running via OpenAI Official API
To run the tagging script using the official OpenAI API leave `--api-url` to its default value. You must specify a model to use (e.g. `gpt-5-mini`). The minimal command becomes:
```
python make_tags.py <hf_path> --model <model> (--tree <tree_path>)
```

## Running via vLLM
You may want to use `gpt-oss-20b` or `gpt-oss-120b`. You may do this locally using `vllm serve`. For example, run `gpt-oss-20b` on port `8085`:

```
vllm serve openai/gpt-oss-20b --async-scheduling --host 0.0.0.0 --port 8085
```

Then leave `--model` to its default empty value but set the `--api-url` to point to the vLLM API. Here is a minimal example:

```
python make_tags.py <hf_path> (--tree <tree_path>) --api-url http://localhost:8085/v1
```

**NOTICE**: As of version `0.11.0` vLLM requires GPU UIDs to be integers starting with 0. In many cases proper GPU IDs are not numeric. To handle this please call:
```
export CUDA_VISIBLE_DEVICES=0,1,...,(NUM_OF_GPUs - 1)
```

## Mechanics
The tagging is based on the `_annotate_single_node` method. That method takes a mathematical text and a node. It generates a tag list consisting of the node's title and the titles of all child nodes of the node. It then instructs the LLM to assign a relevance score to each of the tags in the list. The least relevant tag is get a score of `0.0` and most relevant a score of `1.0`. Remaining scores are assigned accordingly. Then only the names of *child* nodes that surpass a specific threshold (the `--threshold` value) are outputted.

In the entire tagging process this method is first called on the root node and then recursively on each new produces list of nodes until it halts (this is possible beacause the parent's name is also given for tagging, so it is possible to overshadow all of its child nodes) or has covered the entire tree.

The proccess is optimized via a `_TaskPool` that ensures that approx. a fixed number of requests are always handled at a time (currenly fixed to `512`).
