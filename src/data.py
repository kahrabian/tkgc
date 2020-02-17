import math
import numpy as np
import re
from itertools import chain


def split(d, sz):
    num_batches = math.ceil(len(d) / sz)
    return np.array_split(d, num_batches)


def format(d, fs):
    fd = [list(chain.from_iterable([[x[j], ] if f is None else f(x[j]) for j, f in enumerate(fs)])) for x in d]
    return np.array(fd)


def format_time(args, t, t_regx):
    _, m, d, h, _, _ = re.split(t_regx, t)[:-1]  # NOTE: Could use other parts too!
    ft = [f'{m}{d}{h}', ] if args.model == 'TTransE' else [f'{m}m', ] + [f'{x}d' for x in d] + [f'{x}h' for x in h]
    return ft


def index(d):
    return {ent: idx for idx, ent in enumerate(np.unique(d.flatten()))}


def transform(d, idxs):
    td = np.zeros_like(d, dtype=np.int_)
    for i, x in enumerate(d):
        for j, idx in enumerate(idxs):
            if idx is None:
                continue
            td[i][j] = idx[x[j]]
    return td


def _check(x, idx, s, al_ts):
    x = x.copy()
    x[idx] = s
    return tuple(x) in al_ts


def _corrupt(args, pos, al, al_ts, fil):
    neg = np.zeros((pos.shape[0] * args.negative_samples, pos.shape[1]), dtype=np.int_)
    for i in range(args.negative_samples):
        neg[i * pos.shape[0]:(i + 1) * pos.shape[0]] = pos.copy()
        for x in neg[i * pos.shape[0]:(i + 1) * pos.shape[0]]:
            idx = 0 if np.random.random() < 0.5 else 2  # NOTE: Head vs Tail
            ss = al[:, idx].copy()
            s = np.random.choice(ss, 1)[0]
            while s == x[idx] or (fil and _check(x, idx, s, al_ts)):
                s = np.random.choice(ss, 1)[0]
            x[idx] = s
    return neg


def prepare(args, b, al, al_ts, sz, fil, rnd, exp=False):
    pos = b[np.random.choice(b.shape[0], sz)].copy() if rnd else b.copy()
    neg = _corrupt(args, pos, al, al_ts, fil)
    if exp:
        pos = np.expand_dims(pos, axis=2)
        neg = np.expand_dims(neg, axis=2)
    return pos, neg
