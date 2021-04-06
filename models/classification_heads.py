# -*- coding: utf-8 -*-
# @Time : 2019/12/10 15:57
# @Author : Ruiqi Wang

import os
import sys
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction

from inspect import isfunction


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert (A.dim() == 3)
    assert (B.dim() == 3)
    assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1, 2))

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

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    # print(query.size(), support.size())
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    # ************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1, 2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits

    if normalize:
        logits = logits / d

    return logits

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='ProtoNet', enable_scale=True):
        super(ClassificationHead, self).__init__()
        try:
            head = globals()[base_learner]
            self.head = head if isfunction(head) else head()
        except Exception:
            print('No such classification head as {}'.format(base_learner))

        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):

        logits = self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        if self.enable_scale:
            # return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
            return self.scale * logits
        else:
            # return self.head(query, support, support_labels, n_way, n_shot, **kwargs)
            return logits

class ScheduledClassificationHead(nn.Module):
    def __init__(self, base_learner='ProtoNet', enable_scale=True, scale=1.0, fn=None):
        super(ScheduledClassificationHead, self).__init__()
        try:
            head = globals()[base_learner]
            self.head = head if isfunction(head) else head()
        except Exception:
            print('No such classification head as {}'.format(base_learner))

        # Add a learnable scale
        self.enable_scale = enable_scale
        self.fn = fn
        # self.scale = torch.FloatTensor([scale])
        # self.step = torch.FloatTensor([-1])
        self.scale = scale
        self.step = -1

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        logits = self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        if self.enable_scale:
            if self.training:
                self.step += 1
                # self.scale = self.fn(self.scale)
                self.scale = self.fn(self.step)
            # return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
            return self.scale * logits
        else:
            # return self.head(query, support, support_labels, n_way, n_shot, **kwargs)
            return logits

if __name__ == '__main__':
    pass
    # a = locals()['DotHead']
    # print(a)