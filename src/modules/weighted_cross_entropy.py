# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Adapted from fairseq-py

from torch import nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, padding_idx=None, size_average=True, weight=None, eps=0.0):
        super().__init__()
        self.padding_idx = padding_idx
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.eps = eps

    def forward(self, input, target, weight, seq_len, per_loss=False, size_average=True):
        lprobs = F.log_softmax(input, dim=-1)
        target = target.view(-1, 1)

        #print(weight)
        weight = weight.detach()

        nll_loss = -lprobs.gather(dim=-1, index=target)
        weight = weight.repeat(seq_len)
        weight = weight.view(-1, 1)

        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        if self.padding_idx is not None:
            if per_loss:
                pad_mask = target.eq(self.padding_idx)
                nll_loss[pad_mask] = 0 
                return nll_loss
            non_pad_mask = target.ne(self.padding_idx)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]
            weight = weight[non_pad_mask]

        eps_i = self.eps / lprobs.size(-1)
        nll_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        nll_loss = nll_loss * weight

        if per_loss:
            return nll_loss

        if size_average:
            nll_loss = nll_loss.mean()
            return nll_loss
        else:
            sample_size = nll_loss.size(0)
            nll_loss = nll_loss.sum()
            return nll_loss, sample_size

