# -*- coding: utf-8 -*-
# @Time : 2020/3/31 11:01
# @Author : Ruiqi Wang

import torch
from itertools import permutations
from random import shuffle


def mixup(support_data, support_label, query_data, query_label, n_way, new_way):

    T, _, C, H, W = support_data.size()
    sorted_s, indices_s = torch.sort(support_label, dim=1)
    support_data_sorted = torch.stack([support_data[i][indices_s[i]] for i in range(support_data.size(0))], dim=0).\
        view(T, n_way, -1, C, H, W) # T,N,k_s,C,H,W
    # support_label_sorted = torch.stack([support_label[i][indices_s[i]] for i in range(support_data.size(0))], dim=0).view(T,-t)

    sorted_q, indices_q = torch.sort(query_label, dim=1)
    query_data_sorted = torch.stack([query_data[i][indices_q[i]] for i in range(query_data.size(0))], dim=0). \
        view(T, n_way, -1, C, H, W) # T,N,k_q,C,H,W
    # query_label_sorted = torch.stack([query_data[i][indices_s[i]] for i in range(query_data.size(0))], dim=0).view(T,-1)

    indicates1 = list(range(n_way))
    indicates2 = indicates1[1:]+indicates1[:1]

    new_support_data = 0.5*support_data_sorted + 0.5*support_data_sorted[:,indicates2,:,:,:,:]
    new_query_data = 0.5*query_data_sorted + 0.5*query_data_sorted[:,indicates2,:,:,:,:]

    new_support_label = torch.arange(n_way, n_way+new_way).view(1,-1,1).repeat(T,1,support_data_sorted.size(2)).view(T,-1).cuda()
    new_query_label = torch.arange(n_way, n_way+new_way).view(1,-1,1).repeat(T,1,query_data_sorted.size(2)).view(T,-1).cuda()

    support_data_sorted = torch.cat([support_data_sorted, new_support_data], dim=1)
    query_data_sorted = torch.cat([query_data_sorted, new_query_data], dim=1)
    sorted_s = torch.cat([sorted_s, new_support_label], dim=1)
    sorted_q = torch.cat([sorted_q, new_query_label], dim=1)

    return support_data_sorted.view(T,-1,C,H,W), sorted_s, query_data_sorted.view(T,-1,C,H,W), sorted_q


def CF_H(support_data, support_label, query_data, query_label, n_way, new_way):
    T, _, C, H, W = support_data.size()
    sorted_s, indices_s = torch.sort(support_label, dim=1)
    support_data_sorted = torch.stack([support_data[i][indices_s[i]] for i in range(support_data.size(0))], dim=0). \
        view(T, n_way, -1, C, H, W)  # T,N,k_s,C,H,W
    # support_label_sorted = torch.stack([support_label[i][indices_s[i]] for i in range(support_data.size(0))], dim=0).view(T,-t)

    sorted_q, indices_q = torch.sort(query_label, dim=1)
    query_data_sorted = torch.stack([query_data[i][indices_q[i]] for i in range(query_data.size(0))], dim=0). \
        view(T, n_way, -1, C, H, W)

    ps = list(permutations(range(n_way), 2))
    shuffle(ps)
    chosen = ps[:new_way]
    indicates1, indicates2 = zip(*chosen)

    support_data_sorted_top, support_data_sorted_bottom = support_data_sorted.split(H // 2, dim=-2)
    query_data_sorted_top, query_data_sorted_bottom = query_data_sorted.split(H // 2, dim=-2)

    new_support_data = torch.cat([support_data_sorted_top[:, indicates1], support_data_sorted_bottom[:, indicates2]],
                                 dim=-2)
    new_query_data = torch.cat([query_data_sorted_top[:, indicates1], query_data_sorted_bottom[:, indicates2]], dim=-2)

    new_support_label = torch.arange(n_way, n_way + new_way).view(1, -1, 1).repeat(T, 1,
                                                                                   support_data_sorted.size(2)).view(T,
                                                                                                                     -1).cuda()
    new_query_label = torch.arange(n_way, n_way + new_way).view(1, -1, 1).repeat(T, 1, query_data_sorted.size(2)).view(
        T, -1).cuda()

    support_data_sorted = torch.cat([support_data_sorted, new_support_data], dim=1)
    query_data_sorted = torch.cat([query_data_sorted, new_query_data], dim=1)
    sorted_s = torch.cat([sorted_s, new_support_label], dim=1)
    sorted_q = torch.cat([sorted_q, new_query_label], dim=1)

    return support_data_sorted.view(T, -1, C, H, W), sorted_s, query_data_sorted.view(T, -1, C, H, W), sorted_q