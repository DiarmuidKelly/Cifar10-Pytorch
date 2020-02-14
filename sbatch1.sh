#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
#module load binutils/2.31.1-GCCcore-8.2.0
#module load PyTorch
pip install torchvision
pip install scipy

#python ./main.py -c -m 6 -

python ./main.py -p -c -m 13 -ep 4|| true
python ./main.py -p -c -m 14 -ep 4|| true
python ./main.py -p -c -m 15 -ep 4|| true
python ./main.py -p -c -m 16 -ep 4|| true
python ./main.py -p -c -m 23 -ep 4|| true
python ./main.py -p -c -m 24 -ep 4|| true
python ./main.py -p -c -m 25 -ep 4|| true
python ./main.py -p -c -m 26 -ep 4|| true
python ./main.py -p -c -m 27 -ep 4|| true
python ./main.py -p -c -m 28 -ep 4|| true
python ./main.py -p -c -m 29 -ep 4|| true
python ./main.py -p -c -m 1 -ep 4|| true
python ./main.py -p -c -m 2 -ep 4|| true
python ./main.py -p -c -m 3 -ep 4|| true
python ./main.py -p -c -m 4 -ep 4|| true

#jobinfo 9549415
