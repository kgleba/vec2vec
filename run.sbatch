#!/bin/bash
#SBATCH --job-name=train-translator
#SBATCH --error=logs/train-%j.err
#SBATCH --output=logs/train-%j.log
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=40G
#SBATCH --constraint="type_e"

module load Python/Anaconda_v02.2024

# uv pip install numpy==1.26 torch torchvision evaluate vec2text beir toml wandb python-dotenv diffusers faiss-cpu==1.8.0 seaborn
source .venv/bin/activate

nvidia-smi
accelerate launch --multi_gpu --num_processes 2 train.py