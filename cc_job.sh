#!/bin/bash
#SBATCH --account=def-jinguo
#SBATCH --job-name=gg-kge
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

source activate gg

python -BW ignore main.py -ds GitGraph \
                          -m TADistMult \
                          -d 0.5 \
                          -lr 1e-5 \
                          -es 512 \
                          -bs 16384 \
                          -e 1000 \
                          -f \
                          -l 0.1 \
                          -m1 6 \
                          -m2 6 \
                          -s 2020 \
                          -md head \
                          -lf 100
