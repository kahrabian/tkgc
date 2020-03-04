import horovod.torch as hvd
from torch.utils.tensorboard import SummaryWriter

import src.utils as utils


def main():
    args = utils.initialize()
    tr, vd, ts, e_idx_ln, r_idx_ln, t_idx_ln = utils.data(args)
    mdl, opt, ls_f, st_e, bst_ls = utils.prepare(args, e_idx_ln, r_idx_ln, t_idx_ln)
    tb_sw = SummaryWriter() if hvd.rank() == 0 else None
    if not args.test:
        ls_mtr = utils.BestMetric() if hvd.rank() == 0 else None
        for e in range(st_e, args.epochs + 1):
            utils.train(args, e, mdl, opt, ls_f, tr, tb_sw)
            if e % args.log_frequency == 0 or e == args.epochs:
                utils.validate(args, e, mdl, opt, ls_f, vd, ls_mtr, tb_sw)
    else:
        utils.test(args, mdl, ts, tb_sw)
    if hvd.rank() == 0:
        tb_sw.close()


if __name__ == '__main__':
    main()
