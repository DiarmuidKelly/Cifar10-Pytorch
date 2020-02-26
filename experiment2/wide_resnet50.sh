#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12000
module load Python
pip install torchvision

python ./main.py -p -c -m 34 -ep 20 -n || true
python ./main.py -p -c -m 34 -ep 20 -n -pt || true
python ./main.py -p -c -m 34 -ep 20 -mo 0.9 || true
python ./main.py -p -c -m 34 -ep 20 -b 16 || true
python ./main.py -p -c -m 34 -ep 20 -wd 0.9 || true
python ./main.py -p -c -m 34 -ep 20 -lrs -lr 0.002 || true
python ./main.py -p -c -m 34 -ep 20 -op 2 || true
