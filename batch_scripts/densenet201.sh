#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
pip install torchvision

python ./main.py -p -c -m 4 -ep 20 || true
python ./main.py -p -c -m 4 -ep 20 -pt || true
#python ./main.py -p -c -m 4 -ep 20 -mo 0.9 || true
#python ./main.py -p -c -m 4 -ep 20 -b 16 || true
#python ./main.py -p -c -m 4 -ep 20 -n || true
#python ./main.py -p -c -m 4 -ep 20 -n -pt || true
