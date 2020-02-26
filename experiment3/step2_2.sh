#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
pip install torchvision

python ./main.py -p -c -m 16 -ep 20 -n -pt -b 16 || true

