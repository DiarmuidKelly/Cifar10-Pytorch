#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
#module load binutils/2.31.1-GCCcore-8.2.0
#module load PyTorch
pip install torchvision
pip install scipy

#python ./main.py -c -m 6 -

python ./main.py -p -c -m 25 -ep 4|| true
python ./main.py -p -c -m 26 -ep 4|| true
python ./main.py -p -c -m 27 -ep 4|| true
python ./main.py -p -c -m 28 -ep 4|| true
python ./main.py -p -c -m 29 -ep 4|| true
python ./main.py -p -c -m 30 -ep 4|| true
python ./main.py -p -c -m 31 -ep 4|| true
python ./main.py -p -c -m 32 -ep 4|| true


#jobinfo 9549415
