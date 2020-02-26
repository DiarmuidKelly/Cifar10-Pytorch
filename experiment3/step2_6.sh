#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
pip install torchvision

python ./main.py -p -c -m 34 -ep 20 -b 16 -lrs -lr 0.002 || true

