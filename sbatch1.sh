#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
module load Python
#module load binutils/2.31.1-GCCcore-8.2.0
#module load PyTorch

python ./main.py >> test.txt
