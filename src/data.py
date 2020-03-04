import numpy as np
from datetime import datetime
from torch.utils.data import Dataset as tDataset


class Dataset(tDataset):
    def _load(self, fn):
        with open(fn, 'r') as f:
            self._d = np.array(list(map(lambda x: x.split()[:4], f.read().split('\n')[1:-1])), ndmin=3)

    def _format(self):
        self._d = np.apply_along_axis(lambda x: x[:3].tolist() + self._format_time(x[3]), 2, self._d)

    def _format_time(self, t):
        t = datetime.fromtimestamp(int(t))
        d, h = t.day, t.hour  # NOTE: Could use other parts too!
        if self._args.model == 'TTransE':
            ft = [f'{d}{h}', ]
        else:
            ft = [f'{x}d' for x in f'{d:02}'] + [f'{x}h' for x in f'{h:02}']
        return ft

    def transform(self, idx, ts=True, ts_bs=None):
        self._d = np.apply_along_axis(lambda x: x[:3].astype(np.int_).tolist() + [idx[y] for y in x[3:]], 2, self._d)
        self._ts = {**ts_bs, **{tuple(x): True for x in self._d[0, :, :3]}} if ts else {}

    def __array__(self):
        return self._d

    def __init__(self, args, fn, e_idx_ln, md):
        super().__init__()

        self._args = args
        self._e_idx_ln = e_idx_ln
        self._md = md

        self._load(fn)
        self._format()

    def __len__(self):
        return self._d.shape[1]

    def _check(self, x, th, s):
        x = x.copy()
        x[th] = s
        return self._ts.get(tuple(x), False)

    def _corrupt(self, p):
        n = p.copy()
        for i, x in enumerate(n):
            ht = 0 if np.random.random() < 0.5 else 1  # NOTE: Head vs Tail
            s = np.random.randint(0, self._e_idx_ln)
            while s == x[ht] or (self._args.filter and self._check(x, ht, s)):
                s = np.random.randint(0, self._e_idx_ln)
            n[i][ht] = s
        return n

    def _prepare(self, x):
        p = np.repeat(x, self._args.negative_samples if self._args.model != 'TDistMult' else 1, axis=0)
        n = self._corrupt(p)
        return p, n

    def __getitem__(self, i):
        if self._md == 1:
            return self._prepare(self._d[:, i])
        if self._md == 2:
            return self._prepare(self._d[:, i]) + (self._d[:, i],)
        if self._md == 3:
            return self._d[:, i]
