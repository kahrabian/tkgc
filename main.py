import time
import numpy as np
import torch

import src.data as data
import src.utils as utils


def main():
    dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = utils.get_args()

    torch.manual_seed(args.seed)

    tr, vd, ts, tr_ts, e_idx_ln, r_idx_ln, t_idx_ln = utils.get_data(args)

    mdl = utils.get_model(args, e_idx_ln, r_idx_ln, t_idx_ln).to(dvc)
    loss_f = utils.get_loss_f(args).to(dvc)
    optim = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate)

    if not args.test:
        tr_bs = data.split(tr, args.batch_size)
        vd_bs = data.split(vd, args.batch_size)
        for epoch in range(args.epochs):
            np.random.shuffle(tr_bs)

            tr_loss = 0
            st_tm = time.time()
            mdl.train()
            for b in tr_bs:
                loss = utils.get_loss(args, b, tr, tr_ts, mdl, loss_f, dvc)
                loss.backward()
                optim.step()
                tr_loss += loss.item()

            print(f'[{time.time() - st_tm}] Epoch {epoch + 1}/{args.epochs} training loss: {tr_loss / len(tr_bs)}')

            if (epoch + 1) % args.log_frequency == 0 or epoch == (args.epochs - 1):
                vd_loss = 0
                st_tm = time.time()
                mdl.eval()
                for b in vd_bs:
                    loss = utils.get_loss(args, b, tr, tr_ts, mdl, loss_f, dvc)
                    vd_loss += loss.item()

                print(f'[{time.time() - st_tm}] Epoch {epoch + 1}/{args.epochs} validation loss: {vd_loss / len(vd_bs)}')

                utils.save(args, mdl)

    mtr = utils.Metric()
    mdl.eval()
    ts_bs = data.split(ts if args.test else vd, args.batch_size)
    for b in ts_bs:
        utils.evaluate(args, b, mdl, mtr, dvc)
    print(mtr)


if __name__ == '__main__':
    main()
    exit(0)
