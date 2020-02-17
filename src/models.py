import torch
import torch.nn as nn
import torch.nn.functional as F


class TTransE(nn.Module):
    def _score(self, s, o, r):
        return torch.norm(s + r - o, p=1 if self.l1 else 2, dim=1)

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super(TTransE, self).__init__()
        self.l1 = args.l1

        self.e_embed = nn.Embedding(e_cnt, args.embedding_size)
        self.r_embed = nn.Embedding(r_cnt, args.embedding_size)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        self.e_embed.weight.data = F.normalize(self.e_embed.weight, p=2, dim=1)
        self.r_embed.weight.data = F.normalize(self.r_embed.weight, p=2, dim=1)

        self.t_trans = nn.Linear(args.embedding_size, args.embedding_size, bias=False)
        nn.init.uniform_(self.t_trans.weight)

    def forward(self, pos_s, pos_r, pos_o, neg_s, neg_r, neg_o):
        pos_s_e = self.e_embed(pos_s)
        pos_o_e = self.e_embed(pos_o)
        pos_r_e = self.r_embed(pos_r)

        neg_s_e = self.e_embed(neg_s)
        neg_o_e = self.e_embed(neg_o)
        neg_r_e = self.r_embed(neg_r)

        pos = self._score(pos_s_e, pos_o_e, pos_r_e)
        neg = self._score(neg_s_e, neg_o_e, neg_r_e)

        return pos, neg


class TAX(nn.Module):
    def _score(self, s, o, rt):
        raise NotImplementedError(f'this method should be implemented in {self.__class__}')

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super(TAX, self).__init__()
        self.dropout = args.dropout

        self.lstm = nn.LSTM(args.embedding_size, args.embedding_size, num_layers=1, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

        self.e_embed = nn.Embedding(e_cnt, args.embedding_size)
        self.r_embed = nn.Embedding(r_cnt, args.embedding_size)
        self.t_embed = nn.Embedding(t_cnt, args.embedding_size)
        nn.init.xavier_uniform_(self.e_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        nn.init.xavier_uniform_(self.t_embed.weight)
        self.e_embed.weight.data = F.normalize(self.e_embed.weight, p=2, dim=1)
        self.r_embed.weight.data = F.normalize(self.r_embed.weight, p=2, dim=1)
        self.t_embed.weight.data = F.normalize(self.t_embed.weight, p=2, dim=1)

    def rt_embed(self, r, t):
        r_e = self.r_embed(r)

        bs, ts = t.shape[:2]
        t = t.reshape(-1)

        t_e = self.t_embed(t).view(bs, ts, -1)
        s_e = torch.cat((r_e, t_e), 1)
        _, (h, _) = self.lstm(s_e)

        return h.squeeze()

    def forward(self, pos_s, pos_r, pos_o, pos_t, neg_s, neg_r, neg_o, neg_t):
        pos_s_e = self.e_embed(pos_s).squeeze()
        pos_o_e = self.e_embed(pos_o).squeeze()
        pos_rt_e = self.rt_embed(pos_r, pos_t).squeeze()

        neg_s_e = self.e_embed(neg_s).squeeze()
        neg_o_e = self.e_embed(neg_o).squeeze()
        neg_rt_e = self.rt_embed(neg_r, neg_t).squeeze()

        pos_s_e = F.dropout(pos_s_e, p=self.dropout)
        pos_o_e = F.dropout(pos_o_e, p=self.dropout)
        pos_rt_e = F.dropout(pos_rt_e, p=self.dropout)
        neg_s_e = F.dropout(neg_s_e, p=self.dropout)
        neg_o_e = F.dropout(neg_o_e, p=self.dropout)
        neg_rt_e = F.dropout(neg_rt_e, p=self.dropout)

        pos = self._score(pos_s_e, pos_o_e, pos_rt_e)
        neg = self._score(neg_s_e, neg_o_e, neg_rt_e)

        return pos, neg


class TADistMult(TAX):
    def _score(self, s, o, rt):
        return torch.sum(s * o * rt, dim=1)

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super(TADistMult, self).__init__(args, e_cnt, r_cnt, t_cnt)


class TATransE(TAX):
    def _score(self, s, o, rt):
        return torch.norm(s + rt - o, p=2, dim=1)

    def __init__(self, args, e_cnt, r_cnt, t_cnt):
        super(TATransE, self).__init__(args, e_cnt, r_cnt, t_cnt)
