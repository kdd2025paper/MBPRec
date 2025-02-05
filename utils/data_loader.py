import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import warnings
import torch
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set_p = defaultdict(list)  # 为字典提供一个默认的值（相当于初始化字典）
train_user_set_c = defaultdict(list)
train_user_set_v = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)

## 读取文件中的数据
def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

## 统计分类好的数据
def statistics(train_data_p, train_data_c, train_data_v, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data_p[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data_p[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1
    # 构建训练集、测试集和验证集
    for u_id, i_id in train_data_p:
        train_user_set_p[int(u_id)].append(int(i_id))  #[[i1,i2,...],[]]
    for u_id, i_id in train_data_c:
        train_user_set_c[int(u_id)].append(int(i_id))
    for u_id, i_id in train_data_v:
        train_user_set_v[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

## 构建稀疏图
def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))  # rowsum=[29693, 1]，按行求和，相当于Dk, rowsum里的每个元素代表用户购买物品的个数

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 拉成一行向量  [29693, 1]
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 若出现无穷大则置为0，分母为0说明用户未购买  [29693, 1]
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 矩阵对角化，d_mat_inv_sqrt=[29693, 29693]

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)  # 计算Ak^
        return bi_lap.tocoo()

    def _si_norm_lap(adj):  # 未被调用
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()  # cf = [282860, 2]
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items) 扩展列数
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    #[[u,i],[i,u]]
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]   cf_ = [565720, 2]

    vals = [1.] * len(cf_)  # 矩阵的行数
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))  # 分3行来存储
    return _bi_norm_lap(mat)

## 加载数据
def load_data(model_args):
    global args, dataset  # 定义全局变量
    args = model_args
    dataset = args.dataset

    print('reading train and test user-item set ...')
    directory = args.data_path + dataset + '/'
    if args.dataset == "Yelp":
        train_p = read_cf(directory + 'trn_pos.txt')
        train_c = read_cf(directory + 'trn_neutral.txt')
        train_v = read_cf(directory + 'trn_tip.txt')
    else:                                   # 本实验中使用的是Beibei数据集
        train_p = read_cf(directory + 'train.txt') #[[u,i]...]
        train_c = read_cf(directory + 'cart.txt')
        train_v = read_cf(directory + 'pv.txt')
    #print(train_cf)a
    test_cf = read_cf(directory + 'test.txt')


    valid_cf = test_cf
    statistics(train_p, train_c, train_v, valid_cf, test_cf)
    #print(train_user_set)

    print('building the adj mat ...')  # 构建邻接矩阵
    norm_mat_p = build_sparse_graph(train_p)
    norm_mat_c = build_sparse_graph(train_c)
    norm_mat_v = build_sparse_graph(train_v)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    #print(train_user_set_p[0])

    #args.user_sim
    type_num = []
    pkfile = open(directory + "new/type_num_version/type_num011.txt", 'r')
    for line in pkfile:
        popweights = line.split(' ')
        oneuser = [float(popweights[1]), float(popweights[2]), float(popweights[3][:-1])]
        type_num.append(oneuser)

    # type_num = pickle.load(pkfile)
    # type_num=[21716, 3], 每个元素为1*3的向量，里面的值表示每种行为下的次数占比
    # 3个数之和为1，考虑对数据进行了归一化处理
    #pos_pair = torch.IntTensor(pos_pair)
    #print(train_p)
    #print(test_user_set)
    user_dict = {
        'train_user_set_p': train_user_set_p,
        'train_user_set_c': train_user_set_c,
        'train_user_set_v': train_user_set_v,
        'valid_user_set': None,
        'test_user_set': test_user_set,
    }

    #print(test_user_set)
    print('loading over ...')
    #print(norm_mat_p)
    return train_p, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v, torch.tensor(type_num)

