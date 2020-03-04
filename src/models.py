import torch
import torch.nn as nn
import torch.nn.functional as F


class TTransE(nn.Module):
    def _score(self, s, o, r, t):
        return torch.norm(s + r + t - o, p=1 if self.l1 else 2, dim=1)

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


class TAX(nn.Module):
    def _score(self, s, o, rt):
        raise NotImplementedError(f'this method should be implemented in {self.__class__}')

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super().__init__()

        self.dvc = args.dvc
        self.dropout = args.dropout

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
        s_e = F.dropout(self.e_embed(s).to(self.dvc), p=self.dropout, training=self.training)
        o_e = F.dropout(self.e_embed(o).to(self.dvc), p=self.dropout, training=self.training)
        rt_e = F.dropout(self.rt_embed(r, t), p=self.dropout, training=self.training)
        return self._score(s_e, o_e, rt_e)


class TADistMult(TAX):
    def _score(self, s, o, rt):
        return torch.sum(s * o * rt, dim=1)


class TATransE(TAX):
    def _score(self, s, o, rt):
        return torch.norm(s + rt - o, p=2, dim=1)
