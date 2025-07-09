#!/bin/bash
#SBATCH --job-name=train-translator
#SBATCH --error=logs/train-%j.err
#SBATCH --output=logs/train-%j.log
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G
#SBATCH --constraint="type_e"

module load Python/Anaconda_v03.2023

# uv pip install numpy==1.26 torch torchvision evaluate vec2text beir toml wandb python-dotenv diffusers faiss-gpu seaborn
source .venv/bin/activate

nvidia-smi
python3 train.py