#!/bin/bash
#SBATCH --account=def-jinguo
#SBATCH --job-name=gg-tkgc
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=93G
#SBATCH --time=3-0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

source activate gg

horovodrun -np 2 -H localhost:2020 main.py \
       -ds GitGraph \
       -m TADistMult \
       -d 0.2 \
       -es 256 \
       -lr 0.001 \
       -e 100 \
       -bs 4096 \
       -ns 64 \
       -f \
       -o O2 \
       -md head \
       -lf 10 \
       -w 5
