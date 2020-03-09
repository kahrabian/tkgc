# TKGC: Temporal Knowledge Graph Completion

This repository contains the implementation of the following algorithms:

- [x] TTransE
- [x] TA-DistMult
- [x] TA-TransE
- [x] DE-DistMult
- [x] DE-TransE

## Installation

Before installing `horovod` check the steps [here](https://github.com/horovod/horovod#install).

```bash
pip install -r requirements.txt
```

Run the following script to install the necessary requirements for using TPUs.

```bash
VERSION = "xrt==1.15.0"
curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
python pytorch-xla-env-setup.py --version $VERSION
```

## Usage

Sample distributed CPU/GPU training configuration with 2 processes:

```bash
horovodrun -np 2 -H localhost:2 python -BW ignore main.py \
           --dataset GitGraph \
           --model DEDistMult \
           --dropout 0.2 \
           --embedding-size 128 \
           --learning-rate 0.01 \
           --epochs 100 \
           --batch-size 256 \
           --negative-samples 64 \
           --filter \
           --mode head \
           --validation-frequency 2 \
           --threads 2 \
           --workers 1
```

Sample distributed TPU training configuration with 2 processes:

```bash
NPROC=1 python -BW ignore main.py --dataset GitGraph_0.01 \
                                  --model DEDistMult \
                                  --dropout 0.2 \
                                  --embedding-size 128 \
                                  --learning-rate 0.001 \
                                  --epochs 100 \
                                  --batch-size 1024 \
                                  --negative-samples 64 \
                                  --filter \
                                  --mode head \
                                  --validation-frequency 10 \
                                  --tpu \
                                  --threads 4 \
                                  --workers 1
```

You can use the `--aux-cpu` switch to enable mixed CPU training.

To see the list of all available options use the following command:

```bash
python main.py --help
```
