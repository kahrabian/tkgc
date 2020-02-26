import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.data as data
import src.utils as utils


def main():
    dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = utils.get_args()

    torch.manual_seed(args.seed)

    tr, vd, ts, tr_ts, vd_ts, e_idx_ln, r_idx_ln, t_idx_ln = utils.get_data(args)

    mdl = utils.get_model(args, e_idx_ln, r_idx_ln, t_idx_ln).to(dvc)
    loss_f = utils.get_loss_f(args).to(dvc)
    optim = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate)

    tb_sw = SummaryWriter()

    if not args.test:
        tr_bs = data.split(tr, args.batch_size)
        vd_bs = data.split(vd, args.batch_size)
        for epoch in range(args.epochs):
            np.random.shuffle(tr_bs)

            tr_loss = 0
            st_tm = time.time()
            mdl.train()
            with tqdm(total=len(tr_bs), desc=f'Epoch {epoch + 1}/{args.epochs}') as t:
                for i, b in enumerate(tr_bs):
                    loss = utils.get_loss(args, b, tr, tr_ts, mdl, loss_f, dvc)
                    loss.backward()
                    optim.step()
                    tr_loss += loss.item()

                    tb_sw.add_scalars(f'epoch/{epoch}', {'loss': loss.item(), 'mean_loss': tr_loss / (i + 1)}, i)

                    t.set_postfix(loss=f'{tr_loss / (i + 1):.4f}')
                    t.update()

            el_tm = time.time() - st_tm
            tr_loss /= len(tr_bs)
            print(f'Epoch {epoch + 1}/{args.epochs}: training_loss={tr_loss:.4f}, time={el_tm:.4f}')

            tb_sw.add_scalar(f'loss/train', tr_loss, epoch)

            if (epoch + 1) % args.log_frequency == 0 or epoch == (args.epochs - 1):
                vd_loss = 0
                st_tm = time.time()
                mdl.eval()
                for b in vd_bs:
                    loss = utils.get_loss(args, b, np.concatenate([tr, vd]), tr_ts.union(vd_ts), mdl, loss_f, dvc)
                    vd_loss += loss.item()

                el_tm = time.time() - st_tm
                vd_loss /= len(vd_bs)
                print(f'Epoch {epoch + 1}/{args.epochs}: validation_loss={vd_loss:.4f}, time={el_tm:.4f}')

                tb_sw.add_scalar(f'loss/validation', vd_loss, epoch)

                utils.save(args, mdl)

    mtr = utils.Metric()
    mdl.eval()
    ts_bs = data.split(ts if args.test else vd, args.batch_size)
    for b in ts_bs:
        utils.evaluate(args, b, mdl, mtr, dvc)
    print(mtr)

    tb_sw.add_hparams(vars(args), dict(mtr))

    tb_sw.close()


if __name__ == '__main__':
    main()
