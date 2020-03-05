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

## Usage

```bash
horovodrun -np 2 -H localhost:2 python -BW ignore main.py \
           -ds GitGraph \
           -m TADistMult \
           -d 0.2 \
           -es 256 \
           -lr 0.01 \
           -e 100 \
           -bs 256 \
           -ns 64 \
           -f \
           -fp \
           -as \
           -md head \
           -lf 2 \
           -th 2 \
           -w 1
```

To see the list of all available options use the following command:

```bash
python main.py --help
```
