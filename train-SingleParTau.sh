#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

set -e 

# Default output directory if not provided
OUTPUT_DIR=${1:-"outputs/SingleParTau_experiment"}

echo "Starting training for SingleParTau (task: is_tau, 1 epoch, debug mode)..."
uv run python3 mltau/scripts/train.py \
    training.model.name=SingleParTau \
    training.model.task=is_tau \
    training.trainer.max_epochs=1 \
    training.debug_run=False \
    output_dir=$OUTPUT_DIR

echo "Training and inference finished. Generating ROC plots..."
uv run python3 mltau/scripts/plot_roc.py \
    output_dir=$OUTPUT_DIR

echo "All tasks finished. Check $OUTPUT_DIR/plots for results."
