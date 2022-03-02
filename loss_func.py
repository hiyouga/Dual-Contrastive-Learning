import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    ''' Cross-entropy loss with syntactic regularization term and label-smoothing '''

    def __init__(self, opt):
        super(CrossEntropy, self).__init__()
        self.confidence = 1.0 - opt.eps
        self.smoothing = opt.eps
        self.classes = opt.label_class
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.opt = opt
        self.gamma = opt.gamma
        self.keep_ratio = 0.1
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, labels,
                keep_prob=None, wordpiece_mask=None,
                if_mixup=False):
        # outputs: (bs, nc), labels: (bs, ), drop_prob (bs, sl, 1), wordpiece_mask (bs, sl)
        predicts, _, _ = outputs  # predicts: (bs, 2)
        predicts = self.logsoftmax(predicts)
        with torch.no_grad():
            true_dist = torch.zeros_like(predicts)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            if if_mixup:
                true_dist = labels
            else:
                true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)  # ().scatter_(dim, index, src)

        loss_all = -torch.mean(torch.sum(true_dist * predicts, dim=-1))
        if keep_prob is not None:
            text_len = torch.sum(wordpiece_mask, dim=1)  # (bs, )
            keep_num = torch.multiply(text_len, self.keep_ratio)  # (bs, )
            if self.opt.loss_mode == "baseline":
                pass
            elif self.opt.loss_mode == "la":
                loss_all += self.gamma * self.mse_loss(torch.sum(keep_prob, dim=1).squeeze(-1), keep_num)
            elif self.opt.loss_mode == "l1":
                loss_all += self.gamma * torch.mean((torch.sum(torch.abs(keep_prob.squeeze(-1)), dim=1) / text_len))
            elif self.opt.loss_mode == "lal1":
                loss_all += self.gamma * self.mse_loss(torch.sum(keep_prob, dim=1).squeeze(-1), keep_num)
                loss_all += self.gamma * torch.mean((torch.sum(torch.abs(keep_prob.squeeze(-1)), dim=1) / text_len))
            else:
                raise ValueError
        return loss_all

    def kl_divergence(self, p_logit, q_logit):
        # p_logit: [batch, hd]
        # q_logit: [batch, hd]
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)
