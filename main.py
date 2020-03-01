import logging
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.utils as utils


def _logger():
    logging.basicConfig(format='%(message)s', stream=sys.stdout, level=logging.INFO)
    return logging.getLogger()


def main(gpu):
    l = _logger()

    dvc = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    l.info(f'device: {dvc}\n')

    args = utils.get_args()
    l.info('\n'.join(map(lambda x: f'{x[0]}: {x[1]}', sorted(vars(args).items()))) + '\n')

    torch.manual_seed(args.seed)

    tr, vd, ts, e_idx_ln, r_idx_ln, t_idx_ln = utils.get_data(args, gpu)

    mdl = utils.get_model(args, e_idx_ln, r_idx_ln, t_idx_ln, dvc).to(dvc)
    opt = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.resume != '':
        utils.resume(args, mdl, opt)
    ls_f = utils.get_loss_f(args).to(dvc)

    tb_sw = SummaryWriter()

    if not args.test:
        for e in range(args.epochs):
            tr_ls = 0
            st_tm = time.time()
            mdl.train()
            with tqdm(total=len(tr), desc=f'Epoch {e + 1}/{args.epochs}') as t:
                for i, (p, n) in enumerate(tr):
                    p = p.view(-1, p.shape[-1]).to(dvc)
                    n = n.view(-1, n.shape[-1]).to(dvc)

                    ls = utils.get_loss(args, p, n, mdl, ls_f, dvc)
                    ls.backward()
                    opt.step()
                    tr_ls += ls.item()

                    tb_sw.add_scalars(f'epoch/{e}', {'loss': ls.item(), 'mean_loss': tr_ls / (i + 1)}, i)

                    t.set_postfix(loss=f'{tr_ls / (i + 1):.4f}')
                    t.update()

            el_tm = time.time() - st_tm
            tr_ls /= len(tr)
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
                    vd_ls += ls.item()

                el_tm = time.time() - st_tm
                vd_ls /= len(vd)
                l.info(f'Epoch {e + 1}/{args.epochs}: validation_loss={vd_ls:.4f}, time={el_tm:.4f}')

                tb_sw.add_scalar(f'loss/validation', vd_ls, e)

                utils.checkpoint(args, {'mdl': mdl.state_dict(), 'opt': opt.state_dict()})

    mtr = utils.Metric()
    mdl.eval()
    for b in ts:
        b = b.view(-1, b.shape[-1]).to(dvc)
        utils.evaluate(args, b, mdl, mtr, dvc)
    l.info(mtr)

    tb_sw.add_hparams(vars(args), dict(mtr))

    tb_sw.close()


if __name__ == '__main__':
    main(0)
