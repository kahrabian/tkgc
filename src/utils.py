import argparse
import horovod.torch as hvd
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import src.models as models
from src.data import Dataset


def _allreduce(val, name, op):
    return hvd.allreduce(torch.tensor(val), name=name, op=op).item()


class Metric(object):
    def __init__(self):
        self.cnt = 0
        self.h_1 = 0
        self.h_3 = 0
        self.h_10 = 0
        self.mr = 0
        self.mrr = 0

    def allreduce(self):
        self.cnt = _allreduce(self.cnt, 'metric.cnt', hvd.Sum)
        self.h_1 = _allreduce(self.h_1, 'metric.h_1', hvd.Sum)
        self.h_3 = _allreduce(self.h_3, 'metric.h_3', hvd.Sum)
        self.h_10 = _allreduce(self.h_10, 'metric.h_10', hvd.Sum)
        self.mr = _allreduce(self.mr, 'metric.mr', hvd.Sum)
        self.mrr = _allreduce(self.mrr, 'metric.mrr', hvd.Sum)

    def _normalize(self):
        return self.h_1 / self.cnt, self.h_3 / self.cnt, self.h_10 / self.cnt, self.mr / self.cnt, self.mrr / self.cnt

    def __str__(self):
        h_1, h_3, h_10, mr, mrr = self._normalize()
        return f'\nH@1: {h_1}\nH@3: {h_3}\nH@10: {h_10}\nMR: {mr}\nMRR: {mrr}'

    def __iter__(self):
        h_1, h_3, h_10, mr, mrr = self._normalize()
        yield 'metric/H1', h_1
        yield 'metric/H3', h_3
        yield 'metric/H10', h_10
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


class BestMetric(object):
    def __init__(self):
        self.bst = None

    def update(self, mtr):
        is_bst = self.bst is None or mtr < self.bst
        if is_bst:
            self.bst = mtr
        return is_bst, self.bst


def _args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-ds', '--dataset', type=str, required=True)
    argparser.add_argument('-m', '--model', type=str, required=True, choices=['TTransE', 'TADistMult', 'TATransE'])
    argparser.add_argument('-d', '--dropout', type=float, default=0)
    argparser.add_argument('-l1', '--l1', default=False, action='store_true')
    argparser.add_argument('-es', '--embedding-size', type=int, default=128)
    argparser.add_argument('-mr', '--margin', type=int, default=1)
    argparser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    argparser.add_argument('-wd', '--weight-decay', type=float, default=0)
    argparser.add_argument('-e', '--epochs', type=int, default=1000)
    argparser.add_argument('-bs', '--batch-size', type=int, default=512)
    argparser.add_argument('-ns', '--negative-samples', type=int, default=1)
    argparser.add_argument('-f', '--filter', default=True, action='store_true')
    argparser.add_argument('-r', '--resume', type=str, default='')
    argparser.add_argument('-dt', '--deterministic', default=False, action='store_true')
    argparser.add_argument('-fp', '--fp16', default=False, action='store_true')
    argparser.add_argument('-as', '--adasum', default=False, action='store_true')
    argparser.add_argument('-t', '--test', default=False, action='store_true')
    argparser.add_argument('-md', '--mode', type=str, default='both', choices=['head', 'tail', 'both'])
    argparser.add_argument('-lf', '--log-frequency', type=int, default=100)
    argparser.add_argument('-w', '--workers', type=int, default=1)

    args = argparser.parse_args()

    return args


def initialize():
    hvd.init()

    args = _args()

    if args.deterministic:
        torch.manual_seed(hvd.rank())  # NOTE: Reproducability

    torch.set_num_threads(args.workers)

    args.dvc = f'cuda:{hvd.local_rank()}' if torch.cuda.is_available() else 'cpu'
    if args.dvc != 'cpu':
        cudnn.benchmark = not args.deterministic  # NOTE: Reproducability
        cudnn.deterministic = args.deterministic  # NOTE: Reproducability
        torch.cuda.set_device(args.dvc)

    print(f'device: {args.dvc}\n' + '\n'.join(map(lambda x: f'{x[0]}: {x[1]}', sorted(vars(args).items()))))

    return args


def data(args):
    bpath = os.path.join('./data', args.dataset)

    with open(os.path.join(bpath, 'entity2id.txt'), 'r') as f:
        e_idx_ln = int(f.readline().strip())
    with open(os.path.join(bpath, 'relation2id.txt'), 'r') as f:
        r_idx_ln = int(f.readline().strip())

    tr_ds = Dataset(args, os.path.join(bpath, 'train2id.txt'), e_idx_ln)
    vd_ds = Dataset(args, os.path.join(bpath, 'valid2id.txt'), e_idx_ln)
    ts_ds = Dataset(args, os.path.join(bpath, 'test2id.txt'), e_idx_ln, ns=False)

    t_idx = {e: i for i, e in enumerate(np.unique(np.concatenate([tr_ds, vd_ds, ts_ds], axis=1)[0, :, 3:].flatten()))}
    t_idx_ln = len(t_idx)

    tr_ds.transform(t_idx, ts_bs={})
    vd_ds.transform(t_idx, ts_bs=tr_ds._ts)
    ts_ds.transform(t_idx, ts=False)

    tr_smp = DistributedSampler(tr_ds, num_replicas=hvd.size(), rank=hvd.rank())
    vd_smp = DistributedSampler(vd_ds, num_replicas=hvd.size(), rank=hvd.rank())
    ts_smp = DistributedSampler(ts_ds, num_replicas=hvd.size(), rank=hvd.rank())

    tr = DataLoader(tr_ds, batch_size=args.batch_size, sampler=tr_smp, num_workers=args.workers, pin_memory=True)
    vd = DataLoader(vd_ds, batch_size=args.batch_size, sampler=vd_smp, num_workers=args.workers, pin_memory=True)
    ts = DataLoader(ts_ds, batch_size=args.batch_size, sampler=ts_smp, num_workers=args.workers, pin_memory=True)

    return tr, vd, ts, e_idx_ln, r_idx_ln, t_idx_ln


def _model(args, e_cnt, r_cnt, t_cnt):
    return getattr(models, args.model)(args, e_cnt, r_cnt, t_cnt)


def _resume(args, mdl, opt):
    if not os.path.exists(args.resume):
        raise FileNotFoundError('can\'t find the saved model with the given path')
    ckpt = torch.load(args.resume, map_location=args.dvc)
    if hvd.rank() == 0:
        mdl.load_state_dict(ckpt['mdl'])
        opt.load_state_dict(ckpt['opt'])
    return ckpt['e'], ckpt['bst_ls']


def _loss_f(args):
    if args.model == 'TADistMult':
        return nn.BCEWithLogitsLoss(reduction='mean')
    return nn.MarginRankingLoss(args.margin)


def prepare(args, e_idx_ln, r_idx_ln, t_idx_ln):
    mdl = _model(args, e_idx_ln, r_idx_ln, t_idx_ln)

    lr_sc = hvd.local_size() if hvd.nccl_built() else 1 if args.adasum else hvd.size()
    opt = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate * lr_sc, weight_decay=args.weight_decay)

    st_e, bst_ls = _resume(args, mdl, opt) if args.resume != '' else (1, None)

    opt = hvd.DistributedOptimizer(opt,
                                   named_parameters=mdl.named_parameters(),
                                   compression=hvd.Compression.fp16 if args.fp16 else hvd.Compression.none,
                                   op=hvd.Adasum if args.adasum else hvd.Average)

    hvd.broadcast_parameters(mdl.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)

    ls_f = _loss_f(args).to(args.dvc)

    return mdl, opt, ls_f, st_e, bst_ls


def _loss(args, p, n, mdl, loss_f):
    p_s, p_o, p_r, p_t = p[:, 0], p[:, 1], p[:, 2], p[:, 3:].squeeze()
    n_s, n_o, n_r, n_t = n[:, 0], n[:, 1], n[:, 2], n[:, 3:].squeeze()

    if mdl.training:
        mdl.zero_grad()
    s_p, s_n = mdl(p_s, p_o, p_r, p_t, n_s, n_o, n_r, n_t)

    if args.model == 'TADistMult':
        x = torch.cat([s_p, s_n])
        y = torch.cat([torch.ones(s_p.shape), torch.zeros(s_n.shape)]).to(args.dvc)
        loss = loss_f(x, y)
    else:
        loss = loss_f(s_p, s_n, (-1) * torch.ones(s_p.shape + s_n.shape).to(args.dvc))

    return loss


def train(args, e, mdl, opt, ls_f, tr, tb_sw):
    if hvd.rank() == 0:
        tr_ls = 0
        t = tqdm(total=len(tr), desc=f'Epoch {e}/{args.epochs}')

    mdl.train()
    tr.sampler.set_epoch(e)
    for i, (p, n) in enumerate(tr, 1):
        p = p.view(-1, p.shape[-1]).to(args.dvc)
        n = n.view(-1, n.shape[-1]).to(args.dvc)

        ls = _loss(args, p, n, mdl, ls_f)
        ls.backward()
        opt.step()

        if hvd.rank() == 0:
            tr_ls += ls.item()
            tb_sw.add_scalars(f'epoch/{hvd.rank()}/{e}', {'loss': ls.item(), 'mean_loss': tr_ls / i}, i)
            t.set_postfix(loss=f'{tr_ls / i:.4f}')
            t.update()

    if hvd.rank() == 0:
        t.close()
        tr_ls /= len(tr)
        tb_sw.add_scalar(f'loss/{hvd.rank()}/train', tr_ls, e)


def evaluate(args, b, mdl, mtr):
    ts_r, ts_t = b[:, 2], b[:, 3:].squeeze()
    if args.model == 'TTransE':
        rt_embed = mdl.r_embed(ts_r) + mdl.t_embed(ts_t)
    else:
        rt_embed = mdl.rt_embed(ts_r, ts_t)

    if args.mode != 'tail':
        o_embed = mdl.e_embed(b[:, 1]).to(args.dvc)
        if args.model == 'TADistMult':
            ort = rt_embed * o_embed
            s_r = torch.matmul(ort, mdl.e_embed.weight.t()).argsort(dim=1, descending=True).cpu().numpy()
        else:
            ort = o_embed - rt_embed
            s_r = torch.cdist(mdl.e_embed.weight, ort, p=_p(args)).t().argsort(dim=1, descending=True).cpu().numpy()
        for i, s in enumerate(b[:, 0].cpu().numpy()):
            mtr.update(np.argwhere(s_r[i] == s)[0, 0] + 1)

    if args.mode != 'head':
        s_embed = mdl.e_embed(b[:, 0]).to(args.dvc)
        if args.model == 'TADistMult':
            srt = s_embed * rt_embed
            o_r = torch.matmul(srt, mdl.e_embed.weight.t()).argsort(dim=1, descending=True).cpu().numpy()
        else:
            srt = s_embed + rt_embed
            o_r = torch.cdist(srt, mdl.e_embed.weight, p=_p(args)).argsort(dim=1, descending=True).cpu().numpy()
        for i, o in enumerate(b[:, 1].cpu().numpy()):
            mtr.update(np.argwhere(o_r[i] == o)[0, 0] + 1)


def _checkpoint(args, e, mdl, opt, bst_ls, is_bst):
    ckpt = {'e': e, 'mdl': mdl.state_dict(), 'opt': opt.state_dict(), 'bst_ls': bst_ls}

    bpth = os.path.join('./models', args.dataset)
    os.makedirs(bpth, exist_ok=True)

    inc = {'model', 'dropout', 'l1', 'embedding_size', 'margin', 'learning_rate', 'weight_decay', 'negative_samples'}
    fn = '-'.join(map(lambda x: f'{x[0]}_{x[1]}', sorted(filter(lambda x: x[0] in inc, vars(args).items())))) + '.ckpt'
    torch.save(ckpt, os.path.join(bpth, f'e_{e}-' + fn))

    if is_bst:
        shutil.copyfile(os.path.join(bpth, f'e_{e}-' + fn), os.path.join(bpth, f'bst-' + fn))


def validate(args, e, mdl, opt, ls_f, vd, ls_mtr, tb_sw):
    vd_ls = 0
    mdl.eval()
    with torch.no_grad():
        for p, n in vd:
            p = p.view(-1, p.shape[-1]).to(args.dvc)
            n = n.view(-1, n.shape[-1]).to(args.dvc)

            ls = _loss(args, p, n, mdl, ls_f)
            vd_ls += ls.item()

            evaluate(args, b, mdl, mtr)
    vd_ls /= len(vd)
    vd_ls_avg = _allreduce(vd_ls, 'validate.vd_ls_avg', hvd.Average)

    if hvd.rank() == 0:
        print(f'Epoch {e}/{args.epochs} validation loss: {vd_ls_avg}')
        tb_sw.add_scalar(f'loss/{hvd.rank()}/validation', vd_ls_avg, e)

        is_bst, bst_ls = ls_mtr.update(vd_ls_avg)
        _checkpoint(args, e, mdl, opt, bst_ls, is_bst)


def test(args, mdl, ts, tb_sw):
    mtr = Metric()
    mdl.eval()
    with torch.no_grad():
        for b in ts:
            b = b.view(-1, b.shape[-1]).to(args.dvc)
            evaluate(args, b, mdl, mtr)
    mtr.allreduce()

    if hvd.rank() == 0:
        print(mtr)
        tb_sw.add_hparams(vars(args), dict(mtr))
        tb_sw.close()


def _p(args):
    return 1 if args.model == 'TTransE' and args.l1 else 2
