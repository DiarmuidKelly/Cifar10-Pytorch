#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
#module load binutils/2.31.1-GCCcore-8.2.0
#module load PyTorch
pip install torchvision
pip install scipy

python ./main.py -p -c -m 23 -ep 4|| true
python ./main.py -p -c -m 24 -ep 4|| true

#jobinfo 9549415
