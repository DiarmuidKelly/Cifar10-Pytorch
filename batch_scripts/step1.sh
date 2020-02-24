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

python ./main.py -p -c -m 0 -ep 5 -pt|| true
python ./main.py -p -c -m 5 -ep 5 -pt|| true
python ./main.py -p -c -m 15 -ep 5 -pt|| true
python ./main.py -p -c -m 25 -ep 5 -pt|| true
python ./main.py -p -c -m 26 -ep 5 -pt|| true
python ./main.py -p -c -m 31 -ep 5 -pt|| true
python ./main.py -p -c -m 32 -ep 5 -pt|| true
python ./main.py -p -c -m 7 -ep 5 -pt|| true
python ./main.py -p -c -m 10 -ep 5 -pt|| true
python ./main.py -p -c -m 11 -ep 5 -pt|| true



#jobinfo 9549415
