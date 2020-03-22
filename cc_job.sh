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

module load python/3.7.4 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 openmpi/3.1.2 nccl/2.3.5

source ${VENVDIR}/gg/bin/activate

export LD_LIBRARY_PATH=${VENVDIR}/gg/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH}
horovodrun -np 2 -hostfile hostfile --mpi-args="--oversubscribe" --timeline-filename ./logs/timeline-${SLURM_JOB_ID}.json --timeline-mark-cycles python main.py \
           --dataset GitGraph_0.01 \
           --model TADistMult \
           --dropout 0.4 \
           --embedding-size 100 \
           --static-proportion 0.36 \
           --learning-rate 0.01 \
           --learning-rate-step 1 \
           --learning-rate-gamma 1.0 \
           --weight-decay 0.0 \
           --epochs 10 \
           --batch-size 256 \
           --test-batch-size 1 \
           --negative-samples 500 \
           --sampling-technique type \
           --time-fraction 0.0 \
           --filter \
           --fp16 \
           --adasum \
           --mode head \
           --validation-frequency 5 \
           --log-frequency 10 \
           --threads 10 \
           --workers 5
