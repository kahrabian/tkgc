import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractNorm(object):
    def _norm(self, x):
        return torch.norm(x, p=1 if self.l1 else 2, dim=1)


class AbstractDropout(object):
    def _dropout(self, x):
        return F.dropout(x, p=self.dropout, training=self.training)


class AbstractSum(object):
    def _sum(self, x):
        return torch.sum(x, dim=1)


class AbstractTA(nn.Module, AbstractDropout):
    def _score(self, s, o, rt):
        raise NotImplementedError(f'this method should be implemented in {self.__class__}')

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout
        self.l1 = args.l1

        e_es = (2 if args.double_entity_embedding_size else 1) * args.embedding_size
        r_es = (2 if args.double_relation_embedding_size else 1) * args.embedding_size

        self.lstm = nn.LSTM(r_es, r_es, num_layers=1, batch_first=True).to(self.dvc)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.e_embed = nn.Embedding(e_cnt, e_es).to(args.aux_dvc)
        self.r_embed = nn.Embedding(r_cnt, r_es).to(self.dvc)
        self.t_embed = nn.Embedding(t_cnt, r_es).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        nn.init.xavier_uniform_(self.t_embed.weight)

    def _rt_embed(self, r, t):
        r_e = self.r_embed(r)

        bs, ts = t.shape[:2]
        t = t.reshape(-1)

        t_e = self.t_embed(t).view(bs, ts, -1)
        s_e = torch.cat((r_e.unsqueeze(1), t_e), dim=1)
        _, (h, _) = self.lstm(s_e)

        return h.squeeze()

    def forward(self, s, o, r, t):
        s_e = self._dropout(self.e_embed(s).to(self.dvc))
        o_e = self._dropout(self.e_embed(o).to(self.dvc))
        rt_e = self._dropout(self._rt_embed(r, t))
        return self._score(s_e, o_e, rt_e)


class AbstractDE(torch.nn.Module, AbstractDropout):
    def _score(self, st, ot, r):
        raise NotImplementedError(f'this method should be implemented in {self.__class__}')

    def __init__(self, args, e_cnt, r_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout
        self.l1 = args.l1
        self.double_e_es = args.double_entity_embedding_size
        self.double_r_es = args.double_relation_embedding_size

        s_es = int(args.embedding_size * args.static_proportion)
        r_es = args.embedding_size
        t_es = r_es - s_es

        s_es *= 2 if self.double_e_es else 1
        r_es *= 2 if self.double_r_es else 1
        t_es *= 2 if self.double_r_es else 1

        self.e_embed = nn.Embedding(e_cnt, s_es).to(args.aux_dvc)
        self.r_embed = nn.Embedding(r_cnt, r_es).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)

        self.d_frq_embed = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_frq_embed = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        nn.init.xavier_uniform_(self.d_frq_embed.weight)
        nn.init.xavier_uniform_(self.h_frq_embed.weight)

        self.d_phi_embed = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_phi_embed = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        nn.init.xavier_uniform_(self.d_phi_embed.weight)
        nn.init.xavier_uniform_(self.h_phi_embed.weight)

        self.d_amp_embed = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_amp_embed = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        nn.init.xavier_uniform_(self.d_amp_embed.weight)
        nn.init.xavier_uniform_(self.h_amp_embed.weight)

    def _t_embed(self, e, d, h):
        _d = self.d_amp_embed(e).to(self.dvc) * torch.sin(d.view(-1, 1) * self.d_frq_embed(e).to(self.dvc) +
                                                          self.d_phi_embed(e).to(self.dvc))
        _h = self.h_amp_embed(e).to(self.dvc) * torch.sin(h.view(-1, 1) * self.h_frq_embed(e).to(self.dvc) +
                                                          self.h_phi_embed(e).to(self.dvc))
        return _d + _h

    def forward(self, s, o, r, t):
        d, h = t[:, 0].float(), t[:, 1].float()
        s_e = self.e_embed(s).to(self.dvc)
        o_e = self.e_embed(o).to(self.dvc)
        r_e = self.r_embed(r)
        t_s = self._t_embed(s, d, h)
        t_o = self._t_embed(o, d, h)
        if self.double_e_es:
            s_e_r, s_e_i = torch.chunk(s_e, 2, dim=1)
            o_e_r, o_e_i = torch.chunk(o_e, 2, dim=1)
            if self.double_r_es:
                t_s_e, t_s_i = torch.chunk(t_s, 2, dim=1)
                t_o_r, t_o_i = torch.chunk(t_o, 2, dim=1)
                s_t = torch.cat((s_e_r, t_s_e, s_e_i, t_s_i), dim=1)
                o_t = torch.cat((o_e_r, t_o_r, o_e_i, t_o_i), dim=1)
            else:
                s_t = torch.cat((s_e_r, t_s, s_e_i, t_s), dim=1)
                o_t = torch.cat((o_e_r, t_o, o_e_i, t_s), dim=1)
        else:
            s_t = torch.cat((s_e, t_s), dim=1)
            o_t = torch.cat((o_e, t_o), dim=1)
        return self._score(s_t, o_t, r_e)


class TTransE(nn.Module, AbstractNorm):
    def _score(self, s, o, r, t):
        return (-1) * self._norm(s + r + t - o)

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.l1 = args.l1

        self.e_embed = nn.Embedding(e_cnt, args.embedding_size).to(args.aux_dvc)
        self.r_embed = nn.Embedding(r_cnt, args.embedding_size).to(self.dvc)
        self.t_embed = nn.Embedding(t_cnt, args.embedding_size).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        nn.init.xavier_uniform_(self.t_embed.weight)

    def forward(self, s, o, r, t):
        s_e = self.e_embed(s).to(self.dvc)
        o_e = self.e_embed(o).to(self.dvc)
        r_e = self.r_embed(r)
        t_e = self.t_embed(t).squeeze()
        return self._score(s_e, o_e, r_e, t_e)


class TATransE(AbstractTA, AbstractNorm):
    def _score(self, s, o, rt):
        return (-1) * self._norm(s + rt - o)


class DETransE(AbstractDE, AbstractNorm):
    def _score(self, st, ot, r):
        return (-1) * self._norm(self._dropout(st + r - ot))


class TADistMult(AbstractTA, AbstractSum):
    def _score(self, s, o, rt):
        return self._sum(s * o * rt)


class DEDistMult(AbstractDE, AbstractSum):
    def _score(self, st, ot, r):
        return self._sum(self._dropout(st * ot * r))


class TAComplEx(AbstractTA, AbstractSum):
    def _score(self, s, o, rt):
        s_r, s_i = torch.chunk(s, 2, dim=1)
        o_r, o_i = torch.chunk(o, 2, dim=1)
        rt_r, rt_i = torch.chunk(rt, 2, dim=1)

        sc_r = s_r * rt_r * o_r - s_i * rt_i * o_r
        sc_i = s_r * rt_i * o_i + s_i * rt_r * o_i

        return self._sum(torch.cat([sc_r, sc_i], dim=1))


class DEComplEx(AbstractDE, AbstractSum):
    def _score(self, st, ot, r):
        st_r, st_i = torch.chunk(st, 2, dim=1)
        ot_r, ot_i = torch.chunk(ot, 2, dim=1)
        r_r, r_i = torch.chunk(r, 2, dim=1)

        sc_r = st_r * r_r * ot_r - st_i * r_i * ot_r
        sc_i = st_r * r_i * ot_i + st_i * r_r * ot_i

        return self._sum(torch.cat([sc_r, sc_i], dim=1))


class TASimplE(torch.nn.Module, AbstractDropout, AbstractSum):
    def _score(self, s_s, o_s, rt_s, s_o, o_o, rt_o):
        return self._sum(self._dropout((s_s * o_s * rt_s + s_o * o_o * rt_o) / 2.0))

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout

        self.lstm_f = nn.LSTM(args.embedding_size, args.embedding_size, num_layers=1, batch_first=True).to(self.dvc)
        self.lstm_i = nn.LSTM(args.embedding_size, args.embedding_size, num_layers=1, batch_first=True).to(self.dvc)
        for name, param in self.lstm_f.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for name, param in self.lstm_i.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.e_embed_s = nn.Embedding(e_cnt, args.embedding_size).to(args.aux_dvc)
        self.e_embed_o = nn.Embedding(e_cnt, args.embedding_size).to(args.aux_dvc)
        self.r_embed_f = nn.Embedding(r_cnt, args.embedding_size).to(self.dvc)
        self.r_embed_i = nn.Embedding(r_cnt, args.embedding_size).to(self.dvc)
        self.t_embed_f = nn.Embedding(t_cnt, args.embedding_size).to(self.dvc)
        self.t_embed_i = nn.Embedding(t_cnt, args.embedding_size).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed_s.weight)
        nn.init.xavier_uniform_(self.e_embed_o.weight)
        nn.init.xavier_uniform_(self.r_embed_f.weight)
        nn.init.xavier_uniform_(self.r_embed_i.weight)
        nn.init.xavier_uniform_(self.t_embed_f.weight)
        nn.init.xavier_uniform_(self.t_embed_i.weight)

    def _rt_embed(self, r_embed, t_embed, lstm, r, t):
        r_e = r_embed(r)

        bs, ts = t.shape[:2]
        t = t.reshape(-1)

        t_e = t_embed(t).view(bs, ts, -1)
        s_e = torch.cat((r_e.unsqueeze(1), t_e), dim=1)
        _, (h, _) = lstm(s_e)

        return h.squeeze()

    def forward(self, s, o, r, t):
        d, h = t[:, 0].float(), t[:, 1].float()
        s_e_s = self._dropout(self.e_embed_s(s).to(self.dvc))
        o_e_s = self._dropout(self.e_embed_o(o).to(self.dvc))
        s_e_o = self._dropout(self.e_embed_s(o).to(self.dvc))
        o_e_o = self._dropout(self.e_embed_o(s).to(self.dvc))
        rt_e_f = self._dropout(self._rt_embed(self.r_embed_f, self.t_embed_f, self.lstm_f, r, t))
        rt_e_i = self._dropout(self._rt_embed(self.r_embed_i, self.t_embed_i, self.lstm_i, r, t))
        return self._score(s_e_s, o_e_s, rt_e_f, s_e_o, o_e_o, rt_e_i)


class DESimplE(torch.nn.Module, AbstractDropout, AbstractSum):
    def _score(self, st_s, ot_s, r_s, st_o, ot_o, r_o):
        return self._sum(self._dropout((st_s * ot_s * r_s + st_o * ot_o * r_o) / 2.0))

    def __init__(self, args, e_cnt, r_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout

        s_es = int(args.embedding_size * args.static_proportion)
        r_es = args.embedding_size
        t_es = r_es - s_es

        self.e_embed_s = nn.Embedding(e_cnt, s_es).to(args.aux_dvc)
        self.e_embed_o = nn.Embedding(e_cnt, s_es).to(args.aux_dvc)
        self.r_embed_f = nn.Embedding(r_cnt, r_es).to(self.dvc)
        self.r_embed_i = nn.Embedding(r_cnt, r_es).to(self.dvc)
        nn.init.xavier_uniform_(self.e_embed_s.weight)
        nn.init.xavier_uniform_(self.e_embed_o.weight)
        nn.init.xavier_uniform_(self.r_embed_f.weight)
        nn.init.xavier_uniform_(self.r_embed_i.weight)

        self.d_frq_embed_s = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.d_frq_embed_o = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_frq_embed_s = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_frq_embed_o = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        nn.init.xavier_uniform_(self.d_frq_embed_s.weight)
        nn.init.xavier_uniform_(self.d_frq_embed_o.weight)
        nn.init.xavier_uniform_(self.h_frq_embed_s.weight)
        nn.init.xavier_uniform_(self.h_frq_embed_o.weight)

        self.d_phi_embed_s = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.d_phi_embed_o = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_phi_embed_s = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_phi_embed_o = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        nn.init.xavier_uniform_(self.d_phi_embed_s.weight)
        nn.init.xavier_uniform_(self.d_phi_embed_o.weight)
        nn.init.xavier_uniform_(self.h_phi_embed_s.weight)
        nn.init.xavier_uniform_(self.h_phi_embed_o.weight)

        self.d_amp_embed_s = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.d_amp_embed_o = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_amp_embed_s = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        self.h_amp_embed_o = nn.Embedding(e_cnt, t_es).to(args.aux_dvc)
        nn.init.xavier_uniform_(self.d_amp_embed_s.weight)
        nn.init.xavier_uniform_(self.d_amp_embed_o.weight)
        nn.init.xavier_uniform_(self.h_amp_embed_s.weight)
        nn.init.xavier_uniform_(self.h_amp_embed_o.weight)

    def _t_embed(self, e, d, h, md):
        if md == 'S':
            _d = self.d_amp_embed_s(e).to(self.dvc) * torch.sin(d.view(-1, 1) * self.d_frq_embed_s(e).to(self.dvc)
                                                                + self.d_phi_embed_s(e).to(self.dvc))
            _h = self.h_amp_embed_s(e).to(self.dvc) * torch.sin(h.view(-1, 1) * self.h_frq_embed_s(e).to(self.dvc)
                                                                + self.h_phi_embed_s(e).to(self.dvc))
        elif md == 'O':
            _d = self.d_amp_embed_o(e).to(self.dvc) * torch.sin(d.view(-1, 1) * self.d_frq_embed_o(e).to(self.dvc)
                                                                + self.d_phi_embed_o(e).to(self.dvc))
            _h = self.h_amp_embed_o(e).to(self.dvc) * torch.sin(h.view(-1, 1) * self.h_frq_embed_o(e).to(self.dvc)
                                                                + self.h_phi_embed_o(e).to(self.dvc))
        return _d + _h

    def forward(self, s, o, r, t):
        d, h = t[:, 0].float(), t[:, 1].float()
        s_e_s = self.e_embed_s(s).to(self.dvc)
        o_e_s = self.e_embed_o(o).to(self.dvc)
        s_e_o = self.e_embed_s(o).to(self.dvc)
        o_e_o = self.e_embed_o(s).to(self.dvc)
        r_e_f = self.r_embed_f(r)
        r_e_i = self.r_embed_i(r)
        t_e_s_s = self._t_embed(s, d, h, 'S')
        t_e_o_o = self._t_embed(o, d, h, 'O')
        t_e_o_s = self._t_embed(o, d, h, 'S')
        t_e_s_o = self._t_embed(s, d, h, 'O')
        s_t_s_s = torch.cat((s_e_s, t_e_s_s), dim=1)
        o_t_s_o = torch.cat((o_e_s, t_e_o_o), dim=1)
        s_t_o_s = torch.cat((s_e_o, t_e_o_s), dim=1)
        o_t_o_o = torch.cat((o_e_o, t_e_s_o), dim=1)
        return self._score(s_t_s_s, o_t_s_o, r_e_f, s_t_o_s, o_t_o_o, r_e_i)


class TARotatE(AbstractTA, AbstractNorm):
    pi = 3.14159265358979323846

    def _score(self, s, o, rt):
        s_r, s_i = torch.chunk(s, 2, dim=1)
        o_r, o_i = torch.chunk(o, 2, dim=1)

        rt_p = (self.pi * rt) / self.e_r
        rt_r = torch.cos(rt_p)
        rt_i = torch.sin(rt_p)

        sc_r = s_r * rt_r - s_i * rt_i - o_r
        sc_i = s_r * rt_i + s_i * rt_r - o_i

        return (-1) * self._norm(torch.cat([sc_r, sc_i], dim=1))

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__(args, e_cnt, r_cnt, t_cnt)

        self.e_r = 1  # NOTE: LSTM outputs are from [-1, 1]


class DERotatE(AbstractDE, AbstractNorm):
    pi = 3.14159265358979323846

    def _score(self, st, ot, r):
        st_r, st_i = torch.chunk(st, 2, dim=1)
        ot_r, ot_i = torch.chunk(ot, 2, dim=1)

        r_p = (self.pi * r) / self.e_r
        r_r = torch.cos(r_p)
        r_i = torch.sin(r_p)

        sc_r = st_r * r_r - st_i * r_i - ot_r
        sc_i = st_r * r_i + st_i * r_r - ot_i

        return (-1) * self._norm(self._dropout(torch.cat([sc_r, sc_i], dim=1)))

    def __init__(self, args, e_cnt, r_cnt):
        super().__init__(args, e_cnt, r_cnt)

        self.e_r = 6 / np.sqrt((2 if self.double_r_es else 1) * args.embedding_size)
