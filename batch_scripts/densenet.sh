#!/bin/bash
#SBATCH --time=07:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
#module load binutils/2.31.1-GCCcore-8.2.0
#module load PyTorch
pip install torchvision
pip install scipy

python ./main.py -p -c -m 1 -ep 4|| true
python ./main.py -p -c -m 2 -ep 4|| true
python ./main.py -p -c -m 3 -ep 4|| true
python ./main.py -p -c -m 4 -ep 4|| true

