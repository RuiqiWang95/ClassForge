# -*- coding: utf-8 -*-
# @Time : 2019/12/19 9:37
# @Author : Ruiqi Wang

import torch
import torch.nn.functional as F


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


# def protoloss(query_logits, query_labels, n_way, eps=0.0, gamma=0):
def protoloss(query_logits, query_labels, n_way, eps=0.0):
    smoothed_one_hot = one_hot(query_labels.reshape(-1), n_way)
    smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (n_way - 1)

    log_prb = F.log_softmax(query_logits.reshape(-1, n_way), dim=1)
    loss = -smoothed_one_hot * log_prb

    return loss.sum(dim=1)


# def SVMloss(query_logits, query_labels, n_way, eps=0.1, gamma=0):
#     smoothed_one_hot = one_hot(query_labels.reshape(-1), n_way)
#     smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (n_way - 1)
#
#     binary = torch.stack([query_logits, -query_logits], dim=0)
#     log_prb = F.log_softmax(binary, dim=0)[0].reshape(-1, n_way)
#
#     loss = -smoothed_one_hot * log_prb
#     return loss.sum(dim=-1)
#
#
# def focalloss(query_logits, query_labels, n_way, eps=0.1,gamma=0):
#     smoothed_one_hot = one_hot(query_labels.reshape(-1),n_way)
#     ps = torch.softmax(query_logits.reshape(-1, n_way), dim=-1)
#     w = torch.pow(1-ps, gamma)
#     log_prb = F.log_softmax(query_logits.reshape(-1, n_way), dim=1)
#     loss = -smoothed_one_hot * log_prb * w
#
#     return loss.sum(dim=1)
#
#
# def myloss(query_logits, query_labels, n_way, eps=0.1, gamma=0):
#     smoothed_one_hot = one_hot(query_labels.reshape(-1), n_way)
#     ps = torch.softmax(query_logits.reshape(-1, n_way), dim=-1)
#     w = gamma*torch.pow(1 - ps, gamma)
#     log_prb = F.log_softmax(query_logits.reshape(-1, n_way), dim=1)
#     loss = -smoothed_one_hot * log_prb * w
#
#     return loss.sum(dim=1)
