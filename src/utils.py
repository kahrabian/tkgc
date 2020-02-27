import argparse
import numpy as np
import os
import re
import torch
import torch.nn as nn
from itertools import combinations

import src.data as data
import src.models as models


class Metric(object):
    def __init__(self):
        self.cnt = 0
        self.h_1 = 0
        self.h_3 = 0
        self.h_10 = 0
        self.mr = 0
        self.mrr = 0

    def _normalize(self):
        return self.h_1 / self.cnt, self.h_3 / self.cnt, self.h_10 / self.cnt, self.mr / self.cnt, self.mrr / self.cnt

    def __str__(self):
        h_1, h_3, h_10, mr, mrr = self._normalize()
        return f'\nH@1: {h_1}\nH@3: {h_3}\nH@10: {h_10}\nMR: {mr}\nMRR: {mrr}\n'

    def __iter__(self):
        h_1, h_3, h_10, mr, mrr = self._normalize()
        yield 'metric/H@1', h_1
        yield 'metric/H@3', h_3
        yield 'metric/H@10', h_10
        yield 'metric/MR', mr
        yield 'metric/MRR', mrr

    def update(self, r):
        self.cnt += 1

        if r < 2:
            self.h_1 += 1
        if r < 4:
            self.h_3 += 1
        if r < 11:
            self.h_10 += 1

        self.mr += r
        self.mrr += 1.0 / r


def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-ds', '--dataset', type=str, required=True)
    argparser.add_argument('-m', '--model', type=str, required=True, choices=['TTransE', 'TADistMult', 'TATransE'])
    argparser.add_argument('-d', '--dropout', type=float, default=0)
    argparser.add_argument('-l1', '--l1', default=False, action='store_true')
    argparser.add_argument('-es', '--embedding_size', type=int, default=128)
    argparser.add_argument('-mr', '--margin', type=int, default=1)
    argparser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    argparser.add_argument('-e', '--epochs', type=int, default=1000)
    argparser.add_argument('-bs', '--batch_size', type=int, default=512)
    argparser.add_argument('-ns', '--negative_samples', type=int, default=1)
    argparser.add_argument('-f', '--filter', default=True, action='store_true')
    argparser.add_argument('-t', '--test', default=False, action='store_true')
    argparser.add_argument('-md', '--mode', type=str, default='both', choices=['head', 'tail', 'both'])
    argparser.add_argument('-r', '--resume', default=False, action='store_true')
    argparser.add_argument('-s', '--seed', type=int, default=2020)
    argparser.add_argument('-lf', '--log_frequency', type=int, default=100)

    args = argparser.parse_args()

    return args


def _load(pth, fn):
    with open(os.path.join(pth, fn), 'r') as f:
        qs = list(map(lambda x: x.split()[:4], f.read().split('\n')[:-1]))
    return np.array(qs)


def get_data(args):
    bpath = 'data/' + args.dataset
    tr = _load(bpath, 'train.txt')
    vd = _load(bpath, 'valid.txt')
    ts = _load(bpath, 'test.txt')

    qf = [None, None, None, lambda x: data.format_time(args, x, re.compile(r'[-T:Z]'))]
    tr = data.format(tr, qf)
    vd = data.format(vd, qf)
    ts = data.format(ts, qf)

    al = np.concatenate([tr, vd, ts])
    e_idx = data.index(al[:, [0, 2]])
    r_idx = data.index(al[:, 1])
    t_idx = data.index(al[:, 3:])

    e_idx_ln = len(e_idx)
    r_idx_ln = len(r_idx)
    t_idx_ln = len(t_idx)

    qt = [e_idx, r_idx, e_idx] + [t_idx, ] * (1 if args.model == 'TTransE' else 5)
    tr = data.transform(tr, qt)
    vd = data.transform(vd, qt)
    ts = data.transform(ts, qt)

    tr_ts = {tuple(x) for x in tr[:, :3]}
    vd_ts = {tuple(x) for x in vd[:, :3]}

    return tr, vd, ts, tr_ts, vd_ts, e_idx_ln, r_idx_ln, t_idx_ln


def get_p(args):
    bpth = 'models/' + args.dataset
    os.makedirs(bpth, exist_ok=True)

    fn = '-'.join(map(lambda x: f'{x[0]}_{x[1]}', sorted(vars(args).items()))) + '.ckpt'
    p = os.path.join(bpth, fn)
    return p


def get_model(args, e_cnt, r_cnt, t_cnt, dvc):
    p = get_p(args)
    if args.resume:
        if not os.path.exists(p):
            raise FileNotFoundError('can\'t find the saved model with the given params')
        return nn.DataParallel(torch.load(p))
    return nn.DataParallel(getattr(models, args.model)(args, e_cnt, r_cnt, t_cnt, dvc))


def get_loss_f(args):
    if args.model == 'TADistMult':
        return nn.BCEWithLogitsLoss(reduction='mean')
    return nn.MarginRankingLoss(args.margin)


def get_loss(args, b, tr, tr_ts, mdl, loss_f, dvc):
    pos_smp, neg_smp = data.prepare(args, b, tr, tr_ts)

    pos_s = torch.tensor(pos_smp[:, 0]).long().to(dvc)
    pos_r = torch.tensor(pos_smp[:, 1]).long().to(dvc)
    pos_o = torch.tensor(pos_smp[:, 2]).long().to(dvc)
    pos_t = torch.tensor(pos_smp[:, 3:]).long().squeeze().to(dvc)
    neg_s = torch.tensor(neg_smp[:, 0]).long().to(dvc)
    neg_r = torch.tensor(neg_smp[:, 1]).long().to(dvc)
    neg_o = torch.tensor(neg_smp[:, 2]).long().to(dvc)
    neg_t = torch.tensor(neg_smp[:, 3:]).long().squeeze().to(dvc)

    if mdl.training:
        mdl.zero_grad()
    pos, neg = mdl(pos_s, pos_r, pos_o, pos_t, neg_s, neg_r, neg_o, neg_t)

    if args.model == 'TADistMult':
        x = torch.cat([pos, neg])
        y = torch.cat([torch.ones(pos.shape), torch.zeros(neg.shape)]).to(dvc)
        loss = loss_f(x, y)
    else:
        loss = loss_f(pos, neg, (-1) * torch.ones(pos.shape + neg.shape).to(dvc))

    return loss


def _p(args):
    return 1 if args.model == 'TTransE' and args.l1 else 2


def evaluate(args, b, mdl, mtr, dvc):
    ts_r = torch.tensor(b[:, 1]).long().to(dvc)
    ts_t = torch.tensor(b[:, 3:]).long().squeeze().to(dvc)

    if args.model == 'TTransE':
        rt_embed = mdl.module.r_embed(ts_r) + mdl.module.t_embed(ts_t)
    else:
        rt_embed = mdl.module.rt_embed(ts_r, ts_t)

    if args.mode != 'tail':
        ts_o = torch.tensor(b[:, 2]).long().to(dvc)
        o_embed = mdl.module.e_embed(ts_o)
        if args.model == 'TADistMult':
            ort = rt_embed * o_embed
            s_r = torch.matmul(ort, mdl.module.e_embed.weight.t()).argsort(dim=1, descending=True).numpy()
        else:
            ort = o_embed - rt_embed
            s_r = torch.cdist(mdl.module.e_embed.weight, ort, p=_p(args)).t().argsort(dim=1, descending=True).numpy()
        for i, s in enumerate(b[:, 0]):
            mtr.update(np.argwhere(s_r[i] == s)[0, 0] + 1)

    if args.mode != 'head':
        ts_s = torch.tensor(b[:, 0]).long().to(dvc)
        s_embed = mdl.module.e_embed(ts_s)
        if args.model == 'TADistMult':
            srt = s_embed * rt_embed
            o_r = torch.matmul(srt, mdl.module.e_embed.weight.t()).argsort(dim=1, descending=True).numpy()
        else:
            srt = s_embed + rt_embed
            o_r = torch.cdist(srt, mdl.module.e_embed.weight, p=_p(args)).argsort(dim=1, descending=True).numpy()
        for i, o in enumerate(b[:, 2]):
            mtr.update(np.argwhere(o_r[i] == o)[0, 0] + 1)


def save(args, mdl):
    p = get_p(args)
    torch.save(mdl, p)
