import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractNorm(object):
    def _norm(self, x):
        return torch.norm(x, p=1 if self.l1 else 2, dim=1)


class AbstractDropout(object):
    def _dropout(self, x):
        return F.dropout(x, p=self.dropout, training=self.training)


class AbstractTA(nn.Module, AbstractDropout):
    def _score(self, s, o, rt):
        raise NotImplementedError(f'this method should be implemented in {self.__class__}')

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout
        self.l1 = args.l1

        self.lstm = nn.LSTM(args.embedding_size, args.embedding_size, num_layers=1, batch_first=True).to(self.dvc)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.e_embed = nn.Embedding(e_cnt, args.embedding_size).to('cpu' if args.cpu_gpu else self.dvc)
        self.r_embed = nn.Embedding(r_cnt, args.embedding_size).to(self.dvc)
        self.t_embed = nn.Embedding(t_cnt, args.embedding_size).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        nn.init.xavier_uniform_(self.t_embed.weight)

    def rt_embed(self, r, t):
        r_e = self.r_embed(r)

        bs, ts = t.shape[:2]
        t = t.reshape(-1)

        t_e = self.t_embed(t).view(bs, ts, -1)
        s_e = torch.cat((r_e.unsqueeze(1), t_e), 1)
        _, (h, _) = self.lstm(s_e)

        return h.squeeze()

    def forward(self, s, o, r, t):
        s_e = self._dropout(self.e_embed(s).to(self.dvc))
        o_e = self._dropout(self.e_embed(o).to(self.dvc))
        rt_e = self._dropout(self.rt_embed(r, t))
        return self._score(s_e, o_e, rt_e)


class AbstractDE(torch.nn.Module):
    def _score(self, st, ot, r):
        raise NotImplementedError(f'this method should be implemented in {self.__class__}')

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout
        self.l1 = args.l1

        self.e_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        self.r_embed = nn.Embedding(r_cnt, args.embedding_size * 2).cuda()
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)

        self.d_frq_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        self.h_frq_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        nn.init.xavier_uniform_(self.d_frq_embed.weight)
        nn.init.xavier_uniform_(self.h_frq_embed.weight)

        self.d_phi_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        self.h_phi_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        nn.init.xavier_uniform_(self.d_phi_embed.weight)
        nn.init.xavier_uniform_(self.h_phi_embed.weight)

        self.d_amp_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        self.h_amp_embed = nn.Embedding(e_cnt, args.embedding_size).cuda()
        nn.init.xavier_uniform_(self.d_amp_embed.weight)
        nn.init.xavier_uniform_(self.h_amp_embed.weight)

    def _t_embed(self, e, d, h):
        _d = self.d_amp_embed(e) * torch.sin(self.d_frq_embed(e) * d + self.d_phi_embed(e))
        _h = self.h_amp_embed(e) * torch.sin(self.h_frq_embed(e) * h + self.h_phi_embed(e))
        return _d + _h

    def forward(self, s, o, r, t):
        d, h = t[:, 0], t[:, 1]
        s_e = self.e_embed(s)
        o_e = self.e_embed(o)
        r_e = self.r_embed(r)
        t_s = self._t_embed(s, d, h)
        t_o = self._t_embed(o, d, h)
        s_t = torch.cat((s_e, t_s), 1)
        o_t = torch.cat((o_e, t_o), 1)
        return self._score(s_t, o_t, r_e)


class TTransE(nn.Module, AbstractNorm):
    def _score(self, s, o, r, t):
        return self._norm(s + r + t - o)

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.l1 = args.l1

        self.e_embed = nn.Embedding(e_cnt, args.embedding_size).to('cpu' if args.cpu_gpu else self.dvc)
        self.r_embed = nn.Embedding(r_cnt, args.embedding_size).to(self.dvc)
        self.t_embed = nn.Embedding(t_cnt, args.embedding_size).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        nn.init.xavier_uniform_(self.t_embed.weight)

    def forward(self, s, o, r, t):
        s_e = self.e_embed(s).to(self.dvc)
        o_e = self.e_embed(o).to(self.dvc)
        r_e = self.r_embed(r)
        t_e = self.t_embed(t)
        return self._score(s_e, o_e, r_e, t_e)


class TADistMult(AbstractTA):
    def _score(self, s, o, rt):
        return torch.sum(s * o * rt, dim=1)


class TATransE(AbstractTA, AbstractNorm):
    def _score(self, s, o, rt):
        return self._norm(s + rt - o)


class DEDistMult(AbstractDE, AbstractDropout):
    def _score(self, st, ot, r):
        return torch.sum(self._dropout(st * ot * r), dim=1)


class DETransE(AbstractDE, AbstractDropout, AbstractNorm):
    def _score(self, st, ot, r):
        return self._norm(self._dropout(st + r - ot))
