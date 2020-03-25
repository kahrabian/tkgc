import torch.nn as nn
import torch.nn.functional as F


class NegativeSamplingLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.mg = args.margin
        self.ad = args.self_adversarial_sampling
        self.ad_tm = args.self_adversarial_temperature

    def forward(self, p, n, w):
        p_ls = F.logsigmoid(self.mg - p)
        if self.ad:
            n_ls = (F.softmax(self.ad_tm * (n - self.mg), dim=1).detach() * F.logsigmoid(n - self.mg)).sum(dim=1)
        else:
            n_ls = F.logsigmoid(n - self.mg).mean(dim=1)
        return ((-1) * (w * (p_ls + n_ls)).sum() / w.sum()) / 2
