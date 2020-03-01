import logging
import sys
import time
import torch
import torch.distributed as tdist
import torch.nn as nn
try:
    from apex import amp
except ImportError:
    amp = None
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.utils as utils


def _logger():
    logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)
    return logging.getLogger()


def main():
    global args, bst_ls

    if args.local_rank == 0:
        l = _logger()

    dvc = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    if dvc.type == 'cpu':
        if tdist.is_available():
            tdist.init_process_group(backend='gloo')
        elif args.world_size > 1:
            raise RuntimeError('torch.distributed is unavailable, set nproc_per_node to 1')
    else:
        cudnn.benchmark = not args.deterministic  # NOTE: Reproducability
        cudnn.deterministic = args.deterministic  # NOTE: Reproducability

        torch.cuda.set_device(dvc)

        tdist.init_process_group(backend='nccl')

    if args.local_rank == 0:
        l.info('\n'.join(map(lambda x: f'{x[0]}: {x[1]}', sorted(vars(args).items()))) + '\n')

    if args.deterministic:
        torch.manual_seed(args.local_rank)  # NOTE: Reproducability

    tr, vd, ts, e_idx_ln, r_idx_ln, t_idx_ln = utils.get_data(args)

    mdl = utils.get_model(args, e_idx_ln, r_idx_ln, t_idx_ln, dvc)
    if tdist.is_available():
        mdl.to(dvc)
        mdl = DDP(mdl)
    else:
        mdl = nn.DataParallel(mdl)
        mdl.to(dvc)
    opt = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.resume != '':
        st_e, bst_ls = utils.resume(args, mdl, opt, amp, dvc)
    else:
        st_e = 0
    if amp is not None:
        mdl, opt = amp.initialize(mdl, opt, opt_level=args.opt_level)
    ls_f = utils.get_loss_f(args).to(dvc)

    if args.local_rank == 0:
        tb_sw = SummaryWriter()

    if not args.test:
        for e in range(st_e, args.epochs):
            tr_ls = 0
            st_tm = time.time()
            mdl.train()
            if args.local_rank == 0:
                t = tqdm(total=len(tr), desc=f'Epoch {e + 1}/{args.epochs}')
            tr.sampler.set_epoch(e)
            for i, (p, n) in enumerate(tr):
                p = p.view(-1, p.shape[-1]).to(dvc)
                n = n.view(-1, n.shape[-1]).to(dvc)

                ls = utils.get_loss(args, p, n, mdl, ls_f, dvc)
                if amp is not None:
                    with amp.scale_loss(ls, opt) as sc_ls:
                        sc_ls.backward()
                else:
                    ls.backward()
                opt.step()
                if tdist.is_available():
                    ls_item = tdist.all_reduce(ls.clone(), op=tdist.reduce_op.SUM) / args.world_size
                else:
                    ls_item = ls.item()
                tr_ls += ls_item

                if args.local_rank == 0:
                    tb_sw.add_scalars(f'epoch/{e}', {'loss': ls.item(), 'mean_loss': tr_ls / (i + 1)}, i)
                    t.set_postfix(loss=f'{tr_ls / (i + 1):.4f}')
                    t.update()
            if args.local_rank == 0:
                t.close()

            el_tm = time.time() - st_tm
            tr_ls /= len(tr)
            if args.local_rank == 0:
                l.info(f'Epoch {e_idx_ln + 1}/{args.epochs}: training_loss={tr_ls:.4f}, time={el_tm:.4f}')
                tb_sw.add_scalar(f'loss/train', tr_ls, e)

            if (e + 1) % args.log_frequency == 0 or e == (args.epochs - 1):
                vd_ls = 0
                st_tm = time.time()
                mdl.eval()
                for i, (p, n) in enumerate(vd):
                    p = p.view(-1, p.shape[-1]).to(dvc)
                    n = n.view(-1, n.shape[-1]).to(dvc)

                    ls = utils.get_loss(args, p, n, mdl, ls_f, dvc)
                    if tdist.is_available():
                        vd_ls += tdist.all_reduce(ls.clone(), op=tdist.reduce_op.SUM) / args.world_size
                    else:
                        vd_ls += ls.item()

                el_tm = time.time() - st_tm
                vd_ls /= len(vd)
                if args.local_rank == 0:
                    l.info(f'Epoch {e + 1}/{args.epochs}: validation_loss={vd_ls:.4f}, time={el_tm:.4f}')
                    tb_sw.add_scalar(f'loss/validation', vd_ls, e)

                is_bst = bst_ls is None or vd_ls < bst_ls
                if is_bst:
                    bst_ls = vd_ls
                utils.checkpoint(args, e, mdl, opt, amp, bst_ls, is_bst)

    mtr = utils.Metric()
    mdl.eval()
    for b in ts:
        b = b.view(-1, b.shape[-1]).to(dvc)
        utils.evaluate(args, b, mdl, mtr, dvc)
    if args.local_rank == 0:
        l.info(mtr)
        tb_sw.add_hparams(vars(args), dict(mtr))
        tb_sw.close()


if __name__ == '__main__':
    args = utils.get_args()
    bst_ls = None
    main()
