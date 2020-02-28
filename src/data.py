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


def _check(x, idx, s, tr_ts):
    x = x.copy()
    x[idx] = s
    return tr_ts.get(tuple(x), False)


def _corrupt(args, pos, tr, tr_ts):
    neg = pos.copy()
    for i, x in enumerate(neg):
        idx = 0 if np.random.random() < 0.5 else 1  # NOTE: Head vs Tail
        ss = tr[:, idx]  # NOTE: Only choose from existing head/tail space
        s = np.random.choice(ss, 1)[0]  # NOTE: Use while/random instead of for for speed-up
        while s == x[idx] or (args.filter and _check(x, idx, s, tr_ts)):
            s = np.random.choice(ss, 1)[0]
        neg[i][idx] = s
    return neg


def prepare(args, b, tr, tr_ts):
    pos = np.repeat(b, args.negative_samples, axis=0) if args.model != 'TDistMult' else b.copy()
    neg = _corrupt(args, pos, tr, tr_ts)
    return pos, neg
