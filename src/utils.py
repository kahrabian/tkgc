import argparse
import horovod.torch as hvd
import numpy as np
import os
import re
import shutil
import torch
import torch.nn as nn
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pass
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
        return f'H@1: {h_1}\nH@3: {h_3}\nH@10: {h_10}\nMR: {mr}\nMRR: {mrr}'

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


class FakeTimeIndex(object):
    def __len__(self): return 0
    def __getitem__(self, ix): return int(ix)


class Integer(object):
    def __init__(self, val):
        self.val = val


def _args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('-ds', '--dataset', type=str, required=True)
    argparser.add_argument('-mo', '--model', type=str, required=True, choices=['DETransE', 'TATransE', 'TTransE',
                                                                               'DEDistMult', 'TADistMult',
                                                                               'DEComplEx', 'TAComplEx',
                                                                               'DESimplE', 'TASimplE',
                                                                               'DERotatE', 'TARotatE'])
    argparser.add_argument('-do', '--dropout', type=float, default=0)
    argparser.add_argument('-l1', '--l1', default=False, action='store_true')
    argparser.add_argument('-es', '--embedding-size', type=int, default=128)
    argparser.add_argument('-sp', '--static-proportion', type=float, default=1.0)
    argparser.add_argument('-mr', '--margin', type=int, default=1)
    argparser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    argparser.add_argument('-ls', '--learning-rate-step', type=int, default=1)
    argparser.add_argument('-lg', '--learning-rate-gamma', type=float, default=1)
    argparser.add_argument('-wd', '--weight-decay', type=float, default=0)
    argparser.add_argument('-ep', '--epochs', type=int, default=1000)
    argparser.add_argument('-bs', '--batch-size', type=int, default=512)
    argparser.add_argument('-tb', '--test-batch-size', type=int, default=512)
    argparser.add_argument('-ns', '--negative-samples', type=int, default=1)
    argparser.add_argument('-st', '--sampling-technique', type=str, default='random', choices=['random', 'type'])
    argparser.add_argument('-tf', '--time-fraction', type=float, default=0.0)
    argparser.add_argument('-fl', '--filter', default=True, action='store_true')
    argparser.add_argument('-rs', '--resume', type=str, default='')
    argparser.add_argument('-dt', '--deterministic', default=False, action='store_true')
    argparser.add_argument('-fp', '--fp16', default=False, action='store_true')
    argparser.add_argument('-as', '--adasum', default=False, action='store_true')
    argparser.add_argument('-ts', '--test', default=False, action='store_true')
    argparser.add_argument('-md', '--mode', type=str, default='both', choices=['head', 'tail', 'both', 'time'])
    argparser.add_argument('-vf', '--validation-frequency', type=int, default=100)
    argparser.add_argument('-lf', '--log-frequency', type=int, default=100)
    argparser.add_argument('-tp', '--tpu', default=False, action='store_true')
    argparser.add_argument('-ac', '--aux-cpu', default=False, action='store_true')
    argparser.add_argument('-th', '--threads', type=int, default=1)
    argparser.add_argument('-wo', '--workers', type=int, default=1)

    args = argparser.parse_args()

    return args


def _rank(args):
    if args.tpu:
        return xm.get_ordinal()
    return hvd.rank()


def initialize():
    args = _args()

    if not args.tpu:
        hvd.init()

    if args.deterministic:
        torch.manual_seed(_rank(args))  # NOTE: Reproducability

    torch.set_num_threads(args.threads)

    if args.tpu:
        dvc = xm.xla_device()
    elif torch.cuda.is_available():
        cudnn.benchmark = not args.deterministic  # NOTE: Reproducability
        cudnn.deterministic = args.deterministic  # NOTE: Reproducability

        dvc = torch.device(f'cuda:{hvd.local_rank()}')
        torch.cuda.set_device(dvc)
    else:
        dvc = torch.device(f'cpu')

    aux_dvc = torch.device(f'cpu') if args.aux_cpu else dvc

    print(f'device: {dvc.type}\naux_device: {aux_dvc.type}')
    print('\n'.join(map(lambda x: f'{x[0]}: {x[1]}', sorted(vars(args).items()))))

    args.dvc = dvc
    args.aux_dvc = aux_dvc

    args.double_entity_embedding_size = args.model in ['TARotatE', 'DERotatE', 'TAComplEx', 'DEComplEx']
    args.double_relation_embedding_size = args.model in ['TAComplEx', 'DEComplEx']

    return args


def _size(args):
    if args.tpu:
        return xm.xrt_world_size()
    return hvd.size()


def data(args):
    bpath = os.path.join('./data', args.dataset)

    with open(os.path.join(bpath, 'entity2id.txt'), 'r') as f:
        e_ix = dict(map(lambda x: x.split()[::-1], f.read().split('\n')[1:-1]))

    tp_ix, tp_rix = dict(), dict()
    tp_rx = re.compile(r'^/(\w+)/.*$')
    for i, e in e_ix.items():
        tp = re.findall(tp_rx, e)[0]
        if tp not in tp_ix:
            tp_ix[tp] = list()
        tp_ix[tp].append(int(i))
        tp_rix[int(i)] = tp

    with open(os.path.join(bpath, 'entity2id.txt'), 'r') as f:
        e_ix_ln = int(f.readline().strip())
    with open(os.path.join(bpath, 'relation2id.txt'), 'r') as f:
        r_ix_ln = int(f.readline().strip())

    tr_ds = Dataset(args, os.path.join(bpath, 'train2id.txt'), e_ix_ln, tp_ix, tp_rix, 1)
    vd_ds = Dataset(args, os.path.join(bpath, 'valid2id.txt'), e_ix_ln, tp_ix, tp_rix, 2)
    ts_ds = Dataset(args, os.path.join(bpath, 'test2id.txt'), e_ix_ln, tp_ix, tp_rix, 3)

    if args.model.startswith('DE'):
        t_ix = FakeTimeIndex()
    else:
        al_t = np.concatenate([tr_ds, vd_ds, ts_ds], axis=1)[0, :, 3:].flatten()
        t_ix = {e: i for i, e in enumerate(np.unique(al_t))}
    t_ix_ln = len(t_ix)

    tr_ds.transform(t_ix, qs_bs={})
    vd_ds.transform(t_ix, qs_bs=tr_ds._qs)
    ts_ds.transform(t_ix, qs=False)

    tr_smp = DistributedSampler(tr_ds, num_replicas=_size(args), rank=_rank(args))
    vd_smp = DistributedSampler(vd_ds, num_replicas=_size(args), rank=_rank(args), shuffle=False)
    ts_smp = DistributedSampler(ts_ds, num_replicas=_size(args), rank=_rank(args), shuffle=False)

    tr_dl = DataLoader(tr_ds,
                       batch_size=args.batch_size,
                       sampler=tr_smp,
                       num_workers=args.workers,
                       pin_memory=not args.tpu,
                       drop_last=args.tpu)
    vd_dl = DataLoader(vd_ds,
                       batch_size=args.test_batch_size,
                       sampler=vd_smp,
                       num_workers=args.workers,
                       pin_memory=not args.tpu,
                       drop_last=args.tpu)
    ts_dl = DataLoader(ts_ds,
                       batch_size=args.test_batch_size,
                       sampler=ts_smp,
                       num_workers=args.workers,
                       pin_memory=not args.tpu,
                       drop_last=args.tpu)

    return tr_dl, vd_dl, ts_dl, e_ix_ln, r_ix_ln, t_ix_ln, tp_ix, tp_rix


def _model(args, e_cnt, r_cnt, t_cnt):
    if args.model.startswith('DE'):
        return getattr(models, args.model)(args, e_cnt, r_cnt)
    return getattr(models, args.model)(args, e_cnt, r_cnt, t_cnt)


def is_master(args):
    return _rank(args) == 0


def _resume(args, mdl, opt):
    if not os.path.exists(args.resume):
        raise FileNotFoundError('can\'t find the saved model with the given path')
    ckpt = torch.load(args.resume, map_location=args.dvc)
    if is_master(args) or args.tpu:
        mdl.load_state_dict(ckpt['mdl'])
        opt.load_state_dict(ckpt['opt'])
    return ckpt['e'], ckpt['bst_ls']


def _loss_f(args):
    return nn.MarginRankingLoss(args.margin) if args.model == 'TTransE' else nn.CrossEntropyLoss()


def prepare(args, e_ix_ln, r_ix_ln, t_ix_ln):
    mdl = _model(args, e_ix_ln, r_ix_ln, t_ix_ln)

    lr_ml = (hvd.local_size() if hvd.nccl_built() else 1) if not args.tpu and args.adasum else _size(args)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr_ml * args.learning_rate, weight_decay=args.weight_decay)

    st_e, bst_ls = _resume(args, mdl, opt) if args.resume != '' else (1, None)

    if not args.tpu:
        opt = hvd.DistributedOptimizer(opt,
                                       named_parameters=mdl.named_parameters(),
                                       compression=hvd.Compression.fp16 if args.fp16 else hvd.Compression.none,
                                       op=hvd.Adasum if args.adasum else hvd.Average)
        hvd.broadcast_parameters(mdl.state_dict(), root_rank=0)

    lr_sc = torch.optim.lr_scheduler.StepLR(opt, step_size=args.learning_rate_step, gamma=args.learning_rate_gamma)
    if not args.tpu:
        hvd.broadcast_optimizer_state(opt, root_rank=0)

    ls_f = _loss_f(args).to(args.dvc)

    return mdl, opt, lr_sc, ls_f, st_e, bst_ls


def _loss(args, p, n, mdl, loss_f):
    p_s, p_o, p_r, p_t = p[:, 0], p[:, 1], p[:, 2].to(args.dvc), p[:, 3:].to(args.dvc)
    n_s, n_o, n_r, n_t = n[:, 0], n[:, 1], n[:, 2].to(args.dvc), n[:, 3:].to(args.dvc)

    if mdl.training:
        mdl.zero_grad()
    s_p = mdl(p_s, p_o, p_r, p_t)
    s_n = mdl(n_s, n_o, n_r, n_t)

    if args.model == 'TTransE':
        loss = loss_f(s_p, s_n, torch.ones(s_p.shape[0]).to(args.dvc))
    else:
        x = torch.cat((s_p.view(-1, 1), s_n.view(s_p.shape[0], -1)), dim=1)
        y = torch.zeros(s_p.shape[0]).long().to(args.dvc)
        loss = loss_f(x, y)

    return loss


def _train(args, i, tr_ls, ls, tb_sw, e, t):
    n = ((i - 1) % args.log_frequency) + 1
    ls_vl = ls.item()
    tr_ls.val += ls_vl * n
    tb_sw.add_scalars(f'epoch/{e}', {'loss': ls_vl, 'mean_loss': tr_ls.val / i}, i)
    t.set_postfix(loss=f'{tr_ls.val / i:.6f}')
    t.update(n)


def train(args, e, mdl, opt, ls_f, tr_dl, tb_sw):
    if is_master(args):
        tr_ls = Integer(0)
        t = tqdm(total=len(tr_dl), desc=f'Epoch {e}/{args.epochs}')

    tr = pl.ParallelLoader(tr_dl, [args.dvc, ]).per_device_loader(args.dvc) if args.tpu else tr_dl

    mdl.train()
    if not args.tpu:
        tr.sampler.set_epoch(e)
    for i, (p, n) in enumerate(tr, 1):
        p = p.view(-1, p.shape[-1]).to(args.aux_dvc)
        n = n.view(-1, n.shape[-1]).to(args.aux_dvc)

        ls = _loss(args, p, n, mdl, ls_f)
        ls.backward()
        if args.tpu:
            xm.optimizer_step(opt)
        else:
            opt.step()

        if is_master(args) and (i % args.log_frequency == 0 or i == len(tr_dl)):
            if args.tpu:
                xm.add_step_closure(_train, args=(args, i, tr_ls, ls, tb_sw, e, t))
            else:
                _train(args, i, tr_ls, ls, tb_sw, e, t)

    if is_master(args):
        t.close()
        tr_ls.val /= len(tr_dl)
        tb_sw.add_scalar(f'loss/train', tr_ls.val, e)


def _p(args):
    return 1 if (args.model.endswith('TransE') or args.model.endswith('RotatE')) and args.l1 else 2


def _evaluate(args, y_r, y, tp_ix, tp_rix, mtr):
    y_r = y_r.cpu().numpy()
    for i, y_i in enumerate(y.numpy()):
        r = np.argwhere(y_r[i] == y_i[0])[0, 0] + 1
        if args.sampling_technique == 'type':
            r = np.isin(y_r[i], tp_ix[tp_rix[y_i[0]]])[:r].sum()
        mtr.update(r)


def evaluate(args, x, y, mdl, tp_ix, tp_rix, mtr):
    x_s, x_o, x_r, x_t = x[:, 0], x[:, 1], x[:, 2].to(args.dvc), x[:, 3:].to(args.dvc)
    s_x = mdl(x_s, x_o, x_r, x_t).view(args.test_batch_size, -1)

    if args.mode == 'both':
        y_s, y_o = torch.chunk(y, 2, dim=1)
        y_r_s, y_r_o = list(map(lambda x: x.argsort(dim=1, descending=True), torch.chunk(s_x, 2, dim=1)))
        _evaluate(args, y_r_s, y_s, tp_ix, tp_rix, mtr)
        _evaluate(args, y_r_o, y_o, tp_ix, tp_rix, mtr)
    else:
        y_r = s_x.argsort(dim=1, descending=True)
        _evaluate(args, y_r, y, tp_ix, tp_rix, mtr)


def _checkpoint(args, e, mdl, opt, bst_ls, is_bst):
    ckpt = {'e': e, 'mdl': mdl.state_dict(), 'opt': opt.state_dict(), 'bst_ls': bst_ls}

    bpth = os.path.join('./models', args.dataset)
    os.makedirs(bpth, exist_ok=True)

    inc = {'model', 'dropout', 'l1', 'embedding_size', 'margin', 'learning_rate', 'weight_decay', 'negative_samples'}
    fn = '-'.join(map(lambda x: f'{x[0]}_{x[1]}', sorted(filter(lambda x: x[0] in inc, vars(args).items())))) + '.ckpt'
    torch.save(ckpt, os.path.join(bpth, f'e_{e}-' + fn))

    if is_bst:
        shutil.copyfile(os.path.join(bpth, f'e_{e}-' + fn), os.path.join(bpth, f'bst-' + fn))


def _validate(vd_ls, ls):
    vd_ls.val += ls.item()


def validate(args, e, mdl, opt, ls_f, vd_dl, tp_ix, tp_rix, ls_mtr, tb_sw):
    vd_ls = Integer(0)
    mtr = Metric()

    vd = pl.ParallelLoader(vd_dl, [args.dvc, ]).per_device_loader(args.dvc) if args.tpu else vd_dl

    mdl.eval()
    with torch.no_grad():
        for p, n, x, y in vd:
            p = p.view(-1, p.shape[-1]).to(args.aux_dvc)
            n = n.view(-1, n.shape[-1]).to(args.aux_dvc)
            x = x.view(-1, x.shape[-1]).to(args.aux_dvc)

            ls = _loss(args, p, n, mdl, ls_f)
            if args.tpu:
                xm.add_step_closure(_validate, args=(vd_ls, ls))
            else:
                _validate(vd_ls, ls)

            evaluate(args, x, y, mdl, tp_ix, tp_rix, mtr)

    vd_ls.val /= len(vd_dl)
    vd_ls_avg = vd_ls.val if args.tpu else _allreduce(vd_ls.val, 'validate.vd_ls_avg', hvd.Average)
    if not args.tpu:
        mtr.allreduce()

    if is_master(args):
        print(f'Epoch {e}/{args.epochs} validation loss: {vd_ls_avg}')
        print(mtr)
        tb_sw.add_scalar(f'loss/validation', vd_ls_avg, e)
        tb_sw.add_scalars('validation', dict(mtr))

        is_bst, bst_ls = ls_mtr.update(vd_ls_avg)
        _checkpoint(args, e, mdl, opt, bst_ls, is_bst)


def test(args, mdl, ts_dl, tp_ix, tp_rix, tb_sw):
    mtr = Metric()

    ts = pl.ParallelLoader(ts_dl, [args.dvc, ]).per_device_loader(args.dvc) if args.tpu else ts_dl

    mdl.eval()
    with torch.no_grad():
        for x, y in ts:
            x = x.view(-1, x.shape[-1]).to(args.aux_dvc)
            evaluate(args, x, y, mdl, tp_ix, tp_rix, mtr)

    if not args.tpu:
        mtr.allreduce()

    if is_master(args):
        print(mtr)
        del args.dvc, args.aux_dvc  # NOTE: Unsupported type!
        tb_sw.add_hparams(vars(args), dict(mtr))
