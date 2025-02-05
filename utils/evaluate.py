import numpy

from .metrics import *
from .parser import parse_args

import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time


args = parse_args()
Ks = eval(args.Ks) #[20, 40, 60]
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size # 4096


batch_test_flag = args.batch_test_flag # 2048

def test(model, user_dict, n_params, topn):  # main.py中调用，测试时的batch_size=2048

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global test_user_set
    test_user_set = user_dict['test_user_set']

    u_res, i_res = model.generate()

    scores = torch.mm(u_res, i_res.t())            # 给物品打分
    _, rating_K = torch.topk(scores, k=topn)  # 对求出的score取前K项进行排序
    # test = test_list[1]
    c = 0
    ndcg_tem=0
    for i in test_user_set.keys():
        r = [0 for j in range(topn)]  #topK=40
        if int(test_user_set[i][0]) in rating_K[i]:
            rank = rating_K[i].tolist()
            index = rank.index(int(test_user_set[i][0]))
            r[index] = 1
            c = c + 1
        this_ndcg = ndcg_at_k(r, topn, test_user_set[i][0], 1)  # 用的ndcg_at_k
        ndcg_tem=ndcg_tem+this_ndcg
    recall = c / len(test_user_set)
    ndcg = ndcg_tem / len(test_user_set)
 #ndcg_at_k(r, k, ground_truth, method=1)
    return recall,ndcg

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k] #取前k个
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

#    ndcg_at_k(r, K, user_pos_test)
def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """

    GT = set([ground_truth])
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT)) #[1.0]
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

# # zwt代码
# def test(model, user_dict, n_params, topn):  # main.py中调用，测试时的batch_size=2048
#
#     global n_users, n_items
#     n_items = n_params['n_items']
#     n_users = n_params['n_users']
#
#     global test_user_set
#     test_user_set = user_dict['test_user_set']
#
#     u_res, i_res = model.generate()
#
#     scores = torch.mm(u_res, i_res.t())            # 给物品打分
#     _, rating_K = torch.topk(scores, k=topn)  # 对求出的score取前K项进行排序
#     # test = test_list[1]
#     c = 0
#     ndcg_tem=0
#     # print(_[1][1].item())
#     r = numpy.zeros((1, 21716))
#     # print(r[0][6])
#     #r = [0 for j in range(21716)]
#     for i in test_user_set.keys():
#         if int(test_user_set[i][0]) in rating_K[i]:
#             rank = rating_K[i].tolist()
#             index = rank.index(int(test_user_set[i][0]))
#             r[0][i]=_[i][index].item()
#             #r[index] = 1
#             c = c + 1
#     # print(r)
#     this_ndcg = ndcg_at_k(r, topn, 1)  # 用的ndcg_at_k
#     recall = c / len(test_user_set)
#     ndcg = this_ndcg
#  #ndcg_at_k(r, k, ground_truth, method=1)
#     return recall,ndcg

# def dcg_at_k(r, k, method=1):
#     """Score is discounted cumulative gain (dcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Returns:
#         Discounted cumulative gain
#     """
#     r = np.asfarray(r)[:k] #取前k个
#     if r.size:
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.
#
# #    ndcg_at_k(r, K, user_pos_test)
# def ndcg_at_k(r, k, method=1):
#     """Score is normalized discounted cumulative gain (ndcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Returns:
#         Normalized discounted cumulative gain
#
#         Low but correct defination
#     """
#     sent_list=r.sort()
#     sent_list=sent_list[::-1]
#     #sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT)) #[1.0]
#     dcg_max = dcg_at_k(sent_list, k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max

# #gpt给出的ndcg恒为1.0的代码
# def dcg_at_k(r, k, method=1):
#     """
#     计算给定相关性评分列表 r 在前 k 个结果上的 DCG。
#
#     参数:
#         r : list or array-like
#             检索结果的相关性评分列表，按排名顺序排列。
#         k : int
#             要考虑的检索结果数量。
#         method : int (0 or 1)
#             折扣方法的选择，默认为 1。
#
#     返回:
#         float: 前 k 个结果的 DCG 值。
#     """
#     r = np.asfarray(r)[:k]  # 取前 k 个
#     if r.size:
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.
#
#
# def ndcg_at_k(r, k, method=1):
#     """
#     计算给定相关性评分列表 r 在前 k 个结果上的 NDCG。
#
#     参数:
#         r : list or array-like
#             检索结果的相关性评分列表，按排名顺序排列。
#         k : int
#             要考虑的检索结果数量。
#         method : int (0 or 1)
#             折扣方法的选择，默认为 1。
#
#     返回:
#         float: 前 k 个结果的 NDCG 值。
#     """
#     # 创建 r 的副本并对其进行降序排序以计算 IDCG
#     sorted_r = sorted(r, reverse=True)
#     dcg_max = dcg_at_k(sorted_r, k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max
