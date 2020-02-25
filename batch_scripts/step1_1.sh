#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
#module load binutils/2.31.1-GCCcore-8.2.0
#module load PyTorch
pip install torchvision
pip install scipy

#python ./main.py -c -m 6 -

python ./main.py -p -c -m 1 -ep 5 -pt|| true
python ./main.py -p -c -m 4 -ep 5 -pt|| true
python ./main.py -p -c -m 12 -ep 5 -pt|| true
python ./main.py -p -c -m 13 -ep 5 -pt|| true
python ./main.py -p -c -m 16 -ep 5 -pt|| true
python ./main.py -p -c -m 11 -ep 5 -pt|| true
python ./main.py -p -c -m 33 -ep 5 -pt|| true
python ./main.py -p -c -m 34 -ep 5 -pt|| true


#jobinfo 9549415
