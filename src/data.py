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

    def transform(self, t_ix, qs=True, qs_bs=None):
        self._d = np.apply_along_axis(lambda x: x[:3].astype(np.int_).tolist() + [t_ix[y] for y in x[3:]], 2, self._d)
        self._qs = {**qs_bs, **{tuple(x): True for x in self._d[0]}} if qs else {}

        self._t_ix = t_ix
        self._t_ix_ln = len(t_ix)

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
        return self._qs.get(tuple(p_i), False)

    def _corrupt(self, p):
        for p_i in p:
            p_i_c = p_i.copy()
            ix = 0 if np.random.random() < 0.5 else 1  # NOTE: Head vs Tail
            s = np.random.randint(0, self._e_ix_ln)
            while s == p_i[ix] or (self._args.filter and self._check(p_i_c, ix, s)):
                s = np.random.randint(0, self._e_ix_ln)
            p_i[ix] = s

    def _corrupt_type(self, p):
        for p_i in p:
            p_i_c = p_i.copy()
            ix = 0 if np.random.random() < 0.5 else 1  # NOTE: Head vs Tail
            ss = self._tp_ix[self._tp_rix[p_i[ix]]]
            for _ in range(len(ss)):  # NOTE: Heuristic to make the sampling efficient.
                s = ss[np.random.randint(0, len(ss))]
                if s != p_i[ix] and (not self._args.filter or not self._check(p_i_c, ix, s)):
                    break
            else:  # NOTE: Fallback!
                s = np.random.randint(0, self._e_ix_ln)
                while s == p_i[ix] or (self._args.filter and self._check(p_i_c, ix, s)):
                    s = np.random.randint(0, self._e_ix_ln)
            p_i[ix] = s

    def _corrupt_time(self, p):
        for p_i in p:
            p_i_c = p_i.copy()
            if self._args.model == 'TTransE':
                ix = 3
                s = np.random.randint(0, self._t_ix_ln)
                while s == p_i[ix] or (self._args.filter and self._check(p_i_c, ix, s)):
                    s = np.random.randint(0, self._t_ix_ln)
                p_i[ix] = s
            elif self._args.model.startswith('TA'):
                ix = [3, 4, 5, 6]
                s_d = np.random.randint(0, 31)  # NOTE: We assume that each month has 30 days!
                s_h = np.random.randint(0, 24)  # NOTE: Hour
                s = [self._t_ix[f'{x}d'] for x in f'{s_d:02}'] + [self._t_ix[f'{x}h'] for x in f'{s_h:02}']
                while (s == p_i[ix]).all() or (self._args.filter and self._check(p_i_c, ix, s)):
                    s_d = np.random.randint(0, 31)  # NOTE: We assume that each month has 30 days!
                    s_h = np.random.randint(0, 24)  # NOTE: Hour
                    s = [self._t_ix[f'{x}d'] for x in f'{s_d:02}'] + [self._t_ix[f'{x}h'] for x in f'{s_h:02}']
                p_i[ix] = s
            elif self._args.model.startswith('DE'):
                ix = [3, 4]
                s_d = np.random.randint(0, 31)  # NOTE: We assume that each month has 30 days!
                s_h = np.random.randint(0, 24)  # NOTE: Hour
                s = [s_d, s_h]
                while (s == p_i[ix]).all() or (self._args.filter and self._check(p_i_c, ix, s)):
                    s_d = np.random.randint(0, 31)  # NOTE: We assume that each month has 30 days!
                    s_h = np.random.randint(0, 24)  # NOTE: Hour
                    s = [s_d, s_h]
                p_i[ix] = s

    def _prepare_train(self, d):
        p = np.repeat(d, self._args.negative_samples if self._args.model == 'TTransE' else 1, axis=0)
        n = np.repeat(d, self._args.negative_samples, axis=0)

        sf = int(np.ceil(n.shape[0] * (1 - self._args.time_fraction)))
        if self._args.sampling_technique == 'random':
            self._corrupt(n[:sf])
        elif self._args.sampling_technique == 'type':
            self._corrupt_type(n[:sf])
        self._corrupt_time(n[sf:])
        return p, n

    def _prepare_test(self, d):
        y_ix = []
        x = np.repeat(d, self._e_ix_ln * (2 if self._args.mode == 'both' else 1), axis=0)
        if self._args.mode != 'tail':
            y_ix.append(0)
            x[:self._e_ix_ln, 0] = np.arange(self._e_ix_ln)
        if self._args.mode != 'head':
            y_ix.append(1)
            x[-self._e_ix_ln:, 1] = np.arange(self._e_ix_ln)
        y = d[0][y_ix]
        return x, y

    def __getitem__(self, i):
        if self._md == 1:
            return self._prepare_train(self._d[:, i])
        if self._md == 2:
            return self._prepare_train(self._d[:, i]) + self._prepare_test(self._d[:, i])
        if self._md == 3:
            return self._prepare_test(self._d[:, i])
