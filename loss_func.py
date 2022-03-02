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

    def forward(self, outputs, labels):
        predicts, _, _ = outputs  # predicts: (bs, 2)
        predicts = self.logsoftmax(predicts)
        with torch.no_grad():
            true_dist = torch.zeros_like(predicts)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)  # ().scatter_(dim, index, src)

        loss_all = -torch.mean(torch.sum(true_dist * predicts, dim=-1))
        return loss_all

#     def kl_divergence(self, p_logit, q_logit):
#         # p_logit: [batch, hd]
#         # q_logit: [batch, hd]
#         p = F.softmax(p_logit, dim=-1)
#         _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
#         return torch.mean(_kl)
