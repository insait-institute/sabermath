from datasets import load_dataset
import argparse
import yaml

from load_models import ALLOWED_MODELS

from sim_approach0 import calc_approach0_sims
from sim_embeddings import calc_embedding_sims
from sim_jaccard import calc_jaccard_sims
from sim_tfidf import calc_tfidf_sims

parser = argparse.ArgumentParser()
parser.add_argument("--config_file")
parser.add_argument("--method")
parser.add_argument("--force_recalc", action="store_true")
args = parser.parse_args()

config_file_path = args.config_file
method = args.method

if method not in ALLOWED_MODELS + ["jaccard", "approach0", "tf-idf"]:
    raise ValueError(f"Unknown method {method}")

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

fixed_targets_dataset = config["hf_datasets"]["targets_maths_words_fixed"]
fixed_candidates_dataset = config["hf_datasets"]["candidates_maths_words_fixed"]

good_targets = load_dataset(fixed_targets_dataset)["train"]
good_candidates = load_dataset(fixed_candidates_dataset)["train"]

if method in ALLOWED_MODELS:
    calc_embedding_sims(method, good_targets, good_candidates, args.force_recalc)

elif method == "jaccard":
    calc_jaccard_sims(good_targets, good_candidates)

elif method == "approach0":
    calc_approach0_sims(good_targets, good_candidates)

elif method == "tf-idf":
    calc_tfidf_sims(good_targets, good_candidates)
