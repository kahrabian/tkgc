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
        if self._args.model.startswith('DE'):
            ft = [f'{d}', f'{h}']
        elif self._args.model.startswith('TA'):
            ft = [f'{x}d' for x in f'{d:02}'] + [f'{x}h' for x in f'{h:02}']
        else:
            ft = [f'{d}{h}', ]
        return ft

    def transform(self, ix, ts=True, ts_bs=None):
        self._d = np.apply_along_axis(lambda x: x[:3].astype(np.int_).tolist() + [ix[y] for y in x[3:]], 2, self._d)
        self._ts = {**ts_bs, **{tuple(x): True for x in self._d[0, :, :3]}} if ts else {}

    def __array__(self):
        return self._d

    def __init__(self, args, fn, e_ix_ln, tp_ix, tp_rix, md):
        super().__init__()

        self._args = args
        self._e_ix_ln = e_ix_ln
        self._tp_ix = tp_ix
        self._tp_rix = tp_rix
        self._md = md

        self._load(fn)
        self._format()

    def __len__(self):
        return self._d.shape[1]

    def _check(self, p_i, ix, s):
        p_i[ix] = s
        return self._ts.get(tuple(p_i[:3]), False)

    def _corrupt(self, p):
        for i, p_i in enumerate(p):
            p_i = p_i.copy()
            ix = 0 if np.random.random() < 0.5 else 1  # NOTE: Head vs Tail
            s = np.random.randint(0, self._e_ix_ln)
            while s == p[i][ix] or (self._args.filter and self._check(p_i, ix, s)):
                s = np.random.randint(0, self._e_ix_ln)
            p[i][ix] = s

    def _corrupt_type(self, p):
        for i, p_i in enumerate(p):
            p_i = p_i.copy()
            ix = 0 if np.random.random() < 0.5 else 1  # NOTE: Head vs Tail
            ss = self._tp_ix[self._tp_rix[p[i][ix]]]
            for _ in range(len(ss)):  # NOTE: Heuristic to make the sampling efficient.
                s = ss[np.random.randint(0, len(ss))]
                if s != p[i][ix] and (not self._args.filter or not self._check(p_i, ix, s)):
                    break
            else:  # NOTE: Fallback!
                s = np.random.randint(0, self._e_ix_ln)
                while s == p[i][ix] or (self._args.filter and self._check(p_i, ix, s)):
                    s = np.random.randint(0, self._e_ix_ln)
            p[i][ix] = s

    def _prepare(self, x):
        p = np.repeat(x, self._args.negative_samples if self._args.model == 'TTransE' else 1, axis=0)
        n = np.repeat(x, self._args.negative_samples, axis=0)
        if self._args.sampling_technique == 'random':
            self._corrupt(n)
        elif self._args.sampling_technique == 'type':
            self._corrupt_type(n)
        return p, n

    def __getitem__(self, i):
        if self._md == 1:
            return self._prepare(self._d[:, i])
        if self._md == 2:
            return self._prepare(self._d[:, i]) + (self._d[:, i],)
        if self._md == 3:
            return self._d[:, i]
