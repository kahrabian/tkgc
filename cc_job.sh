#!/bin/bash
#SBATCH --account=def-jinguo
#SBATCH --job-name=gg-tkgc
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=93G
#SBATCH --time=3-0
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

source activate gg

python -BW ignore main.py -ds GitGraph \
                          -m TADistMult \
                          -d 0.5 \
                          -es 512 \
                          -mr 6 \
                          -lr 1e-5 \
                          -e 1000 \
                          -bs 16384 \
                          -ns 2 \
                          -f \
                          -md head \
                          -s 2020 \
                          -lf 100
