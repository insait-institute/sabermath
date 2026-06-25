# Anotation Postprocessing

Anotation postprocessing happens in two steps:

## Step 1: Postprocess the databank
Using `./post_process_datasets.py`. This step combines the outputs of the tag and idea anotations, removes problems with empty core ideas or tags lists, and clears redundant tags. The output is a HuggingFace dataset. Usage:

```
python post_process_datasets.py HF_TAGS HF_IDEAS HF_OUT (--tags-column TAG_COL) (--idea-column IDEA_COL) (--make-private)
```

Here the arguments are:

- `HF_TAGS` - HuggingFace path of the tagging output
- `HF_IDEAS` - HuggingFace path of the idea extraction output
- `HF_OUT` - Output HuggingFace path
- `--tags-column TAG_COL` - Name of the column in `HF_TAGS` containing the tag lists (Default: `tags`)
- `--idea-column IDEA_COL` - Name of the column in `HF_IDEAS` containing the ideas (Default: `idea`)
- `--make-private` - Make the output dataset private


## Step 2: Postprocess the tree
Using `./extract_new_tree.py`. This step "composes" the tree from the tags in the tagging script output. Essentially it leaves only tags met at least once in the tagging process, assignes `frequency` field (`0.0-1.0`) and ID to each node. The output is a local JSON file. Usage:

```
python extract_new_tree.py HF_TAGS TREE_OUT (--tags-column TAG_COL)
```

Here the arguments are:

- `HF_TAGS` - HuggingFace path of the tagging output
- `TREE_OUT` - Local output path for the postprocessed JSON tree
- `--tags-column` - Name of the column in `HF_TAGS` containing the tag lists (Default: `tags`)
