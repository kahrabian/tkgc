# TKGC: Temporal Knowledge Graph Completion

This repository contains the implementation of the following algorithms:

- [x] TA-DistMult
- [x] TA-TransE
- [x] TTransE

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -BW ignore main.py -f -ds ICEWS2018 -bs 128 -e 10 -lf 10 -m TADistMult
```

To see the list of all available options use the following command:

```bash
python main.py --help
```
