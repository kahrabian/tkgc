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
        self.mr = 0
        self.mrr = 0
        self.h_1 = 0
        self.h_3 = 0
        self.h_10 = 0

    def __str__(self):
        mr = self.mr / self.cnt
        mrr = self.mrr / self.cnt
        h_1 = self.h_1 / self.cnt
        h_3 = self.h_3 / self.cnt
        h_10 = self.h_10 / self.cnt
        return f'H@1: {h_1}\nH@3: {h_3}\nH@10: {h_10}\nMR: {mr}\nMRR: {mrr}'

    def update(self, r):
        self.cnt += 1

        self.mr += r
        self.mrr += 1.0 / r

        if r < 2:
            self.h_1 += 1
        if r < 4:
            self.h_3 += 1
        if r < 11:
            self.h_10 += 1


def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-ds', '--dataset', type=str, required=True)
    argparser.add_argument('-m', '--model', type=str, required=True, choices=['TTransE', 'TADistMult', 'TATransE'])
    argparser.add_argument('-d', '--dropout', type=float, default=0)
    argparser.add_argument('-l1', '--l1', default=False, action='store_true')
    argparser.add_argument('-es', '--embedding_size', type=int, default=128)
    argparser.add_argument('-l', '--lmbda', type=float, default=0.01)
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


def _load(pth, *fns):
    qs = []
    for fn in fns:
        with open(os.path.join(pth, fn), 'r') as f:
            qs += list(map(lambda x: x.split()[:4], f.read().split('\n')[:-1]))
    return np.array(qs)


def get_data(args):
    bpath = 'data/' + args.dataset
    tr = _load(bpath, 'train.txt')
    vd = _load(bpath, 'valid.txt')
    ts = _load(bpath, 'test.txt')
    al = _load(bpath, 'train.txt', 'valid.txt', 'test.txt')  # NOTE: Train data might be incomplete!

    qf = [None, None, None, lambda x: data.format_time(args, x, re.compile(r'[-T:Z]'))]
    tr = data.format(tr, qf)
    vd = data.format(vd, qf)
    ts = data.format(ts, qf)
    al = data.format(al, qf)

    e_idx = data.index(al[:, [0, 2]])
    r_idx = data.index(al[:, 1])
    t_idx = data.index(al[:, 3:])

    qt = [e_idx, r_idx, e_idx] + [t_idx, ] * (1 if args.model == 'TTransE' else 5)
    tr = data.transform(tr, qt)
    vd = data.transform(vd, qt)
    ts = data.transform(ts, qt)

    tr_ts = {tuple(x) for x in tr[:, :3]}

    e_idx_ln = len(e_idx)
    r_idx_ln = len(r_idx)
    t_idx_ln = len(t_idx)

    return tr, vd, ts, tr_ts, e_idx_ln, r_idx_ln, t_idx_ln


def get_p(args):
    bpth = 'models/' + args.dataset
    os.makedirs(bpth, exist_ok=True)

    fn = '-'.join(map(lambda x: f'{x[0]}_{x[1]}', sorted(vars(args).items()))) + '.ckpt'
    p = os.path.join(bpth, fn)
    return p


def get_model(args, e_cnt, r_cnt, t_cnt):
    p = get_p(args)
    if args.resume:
        if not os.path.exists(p):
            raise FileNotFoundError('can\'t find the saved model with the given params')
        return nn.DataParallel(torch.load(p))
    return nn.DataParallel(getattr(models, args.model)(args, e_cnt, r_cnt, t_cnt))


def get_loss_f(args):
    if args.model == 'TADistMult':
        return nn.BCEWithLogitsLoss(reduction='mean')
    return nn.MarginRankingLoss(args.margin)


def get_reg_f(args, dvc):
    if args.model == 'TTransE':
        return lambda x: torch.sum(torch.max(torch.sum(x ** 2, dim=1) - torch.tensor(1.0), torch.tensor(0.0))).to(dvc)
    return lambda x: torch.mean(x ** 2).to(dvc)


def get_loss(args, b, tr, tr_ts, mdl, loss_f, reg_f, dvc):
    pos_smp, neg_smp = data.prepare(args, b, tr, tr_ts)

    pos_s = torch.LongTensor(pos_smp[:, 0]).to(dvc)
    pos_r = torch.LongTensor(pos_smp[:, 1]).to(dvc)
    pos_o = torch.LongTensor(pos_smp[:, 2]).to(dvc)
    pos_t = torch.LongTensor(pos_smp[:, 3:]).squeeze().to(dvc)
    neg_s = torch.LongTensor(neg_smp[:, 0]).to(dvc)
    neg_r = torch.LongTensor(neg_smp[:, 1]).to(dvc)
    neg_o = torch.LongTensor(neg_smp[:, 2]).to(dvc)
    neg_t = torch.LongTensor(neg_smp[:, 3:]).squeeze().to(dvc)

    if mdl.training:
        mdl.zero_grad()
    pos, neg = mdl(pos_s, pos_r, pos_o, pos_t, neg_s, neg_r, neg_o, neg_t)

    e_embed = mdl.module.e_embed(torch.cat([pos_s, pos_o, neg_s, neg_o]))
    regu = reg_f(e_embed)

    if args.model == 'TTransE':
        r_embed = mdl.module.r_embed(torch.cat([pos_r, neg_r]))
        t_embed = mdl.module.t_embed(torch.cat([pos_t, neg_t]))
        regu += reg_f(r_embed) + reg_f(t_embed)
    else:
        rt_embed = mdl.module.rt_embed(torch.cat([pos_r, neg_r]), torch.cat([pos_t, neg_t]))
        regu += reg_f(rt_embed)

    if args.model == 'TADistMult':
        x = torch.cat([pos, neg])
        y = torch.cat([torch.ones(pos.shape), torch.zeros(neg.shape)]).to(dvc)
        loss = loss_f(x, y)
    else:
        loss = loss_f(pos, neg, (-1) * torch.ones(pos.shape + neg.shape))
    loss += args.lmbda * regu

    return loss


def _p(args):
    return 1 if args.model == 'TTransE' and args.l1 else 2


def evaluate(args, b, mdl, mtr, dvc):
    ts_r = torch.LongTensor(b[:, 1:2]).to(dvc)
    ts_t = torch.LongTensor(b[:, 3:]).to(dvc)

    if args.model == 'TTransE':
        rt_embed = mdl.module.r_embed(ts_r).squeeze() + mdl.module.t_embed(ts_t).squeeze()
    else:
        rt_embed = mdl.module.rt_embed(ts_r, ts_t)

    if args.mode != 'tail':
        ts_o = torch.LongTensor(b[:, 2]).to(dvc)
        o_embed = mdl.module.e_embed(ts_o)
        if args.model == 'TADistMult':
            ort = rt_embed * o_embed
            s_r = torch.matmul(ort, mdl.module.e_embed.weight.t()).argsort(dim=1, descending=True).numpy()
        else:
            ort = o_embed - rt_embed
            s_r = torch.cdist(mdl.module.e_embed.weight, ort, p=_p(args)).argsort(dim=1, descending=True).numpy().T
        for i, s in enumerate(b[:, 0]):
            mtr.update(np.searchsorted(s_r[i], s) + 1)

    if args.mode != 'head':
        ts_s = torch.LongTensor(b[:, 0]).to(dvc)
        s_embed = mdl.module.e_embed(ts_s)
        if args.model == 'TADistMult':
            srt = s_embed * rt_embed
            o_r = torch.matmul(srt, mdl.module.e_embed.weight.t()).argsort(dim=1, descending=True).numpy()
        else:
            srt = s_embed + rt_embed
            o_r = torch.cdist(srt, mdl.module.e_embed.weight, p=_p(args)).argsort(dim=1, descending=True).numpy()
        for i, o in enumerate(b[:, 2]):
            mtr.update(np.searchsorted(o_r[i], o) + 1)


def save(args, mdl):
    p = get_p(args)
    torch.save(mdl, p)
