#!/bin/bash
#SBATCH --job-name=tagging
#SBATCH --nodes=1
#SBATCH --partition=batch       # Specify the partition
#SBATCH --constraint=type-gpu   # Or type-gpu for GPU nodes
#SBATCH --mem=120G              # Memory limit
#SBATCH --mail-type=ALL         # Send updates via Slack
#SBATCH --time=4-0		          # Time limit

#SBATCH --output=/scratch/%u/logs/tags.log
#SBATCH --error=/scratch/%u/logs/tags.error
#SBATCH --gpus h200:8

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch make_tags.slurm <huggingface path>"
  exit 1
fi

user="$USER"
HF_PATH=$1

rsync -aHzv /home/${user}/rank-embedding-math/informal /scratch/${user}/
cd /scratch/${user}/informal/annotate

# Setup HF token
source ~/.keys/huggingface.sh

num_gpus=$(echo "$SLURM_GPUS_ON_NODE" | grep -o '[0-9]*' | head -1)
num_gpus="${num_gpus:-1}"
cuda_devices=$(seq -s, 0 $((num_gpus - 1)))

# Set the CUDA_VISIBLE_DEVICES in vLLM-friendly format
export CUDA_VISIBLE_DEVICES="$cuda_devices"

mkenv -n annotate -f environment.yml -- \
bash ./scripts/make_tags.sh "$num_gpus" 8084 "$HF_PATH"
