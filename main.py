import horovod.torch as hvd
import os
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    pass
from torch.utils.tensorboard import SummaryWriter

import src.utils as utils


def main(ix):
    args = utils.initialize()
    tr_dl, vd_dl, ts_dl, e_ix_ln, r_ix_ln, t_ix_ln = utils.data(args)
    mdl, opt, lr_sc, ls_f, st_e, bst_ls = utils.prepare(args, e_ix_ln, r_ix_ln, t_ix_ln)
    tb_sw = SummaryWriter() if utils.is_master(args) else None
    if not args.test:
        ls_mtr = utils.BestMetric() if utils.is_master(args) else None
        for e in range(st_e, args.epochs + 1):
            utils.train(args, e, mdl, opt, ls_f, tr_dl, tb_sw)
            if e % args.validation_frequency == 0 or e == args.epochs:
                utils.validate(args, e, mdl, opt, ls_f, vd_dl, ls_mtr, tb_sw)
            lr_sc.step()
    else:
        utils.test(args, mdl, ts_dl, tb_sw)
    if utils.is_master(args):
        tb_sw.flush()
        tb_sw.close()


if __name__ == '__main__':
    nproc = os.getenv('NPROC', None)
    if nproc is not None:
        xmp.spawn(main, nprocs=int(nproc))
    else:
        main(0)
