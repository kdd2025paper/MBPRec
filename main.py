import os
import random

import torch
import numpy as np

## 导入utils包里的文件，因为要用到里面的函数来实现功能
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

import datetime

now = datetime.datetime.now()
fname = now.strftime('result/ %Y-%m-%d-%H-%M-%S') + '-multpop' + '.txt'

"""

n_items = 7977   (物品个数)
n_users = 21716  (用户个数)

"""

def main():
    def get_feed_dict(train_p, type_num, ui_dict, start, end, n_negs=1):

        def sampling(user_item, train_set, n):  # 对数据进行采样
            neg_items = []
            for user, _ in user_item.cpu().numpy():
                user = int(user)
                negitems = []  # 保存与用户没有交互过的物品
                for i in range(n):  # sample n times ==> 这里的n = 64，表示从与用户没有交互的项目中选出64个
                    while True:
                        negitem = random.choice(range(n_items)) # 随机选择
                        if negitem not in train_set[user]:
                            break
                    negitems.append(negitem)
                neg_items.append(negitems)
            return neg_items

        ## 将数据取出并封装到字典里
        feed_dict = {}
        entity_pairs = train_p[start:end]  # entity_pairs尺寸 = [4096, 2]
        user = entity_pairs[:, 0]
        type_n = torch.tensor(type_num[user]).t()

        feed_dict['users'] = entity_pairs[:, 0]  # 第一列存用户信息
        feed_dict['type_n'] = type_n
        # print(feed_dict['users'])
        feed_dict['pos_items'] = entity_pairs[:, 1]  # 第二列存与用户有关的物品信息
        feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                           ui_dict,
                                                           n_negs * args.K)).to(device)  # n_negs=64, args.K=5(固定参数)
        return feed_dict

    ## 固定随机数种子，用来保证每一次运行产生的随机数序列都相同
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## 读取参数（调用parser.py文件中parse_args()函数）
    """read args"""
    global args, device
    args = parse_args()

    f = open(fname, 'a')
    namespace_dict = vars(args)
    for key, value in namespace_dict.items():
        f.write(f"{key}: {value}" + '\n')
    f.close()

    ## 指定用作计算的硬件设备（有gpu则使用gpu）popweights[1],
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    #device =torch.device("cpu")

    ## 构建数据集（调用data_loader.py文件中load_data()函数）
    """build dataset"""
    #train_p, train_c, train_v, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v
    train_p, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v, type_num = load_data(args)
    # train_p=[282860, 2], norm_mat_p, norm_mat_c, norm_mat_v=[29693, 29693], type_num=[21716, 3]

    train_p = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_p], np.int32)) #转tensor

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs

    ## 定义LightGCN模型
    """define model"""
    from modules.LightGCN import LightGCN
    #from modules.NGCF import NGCF

    model = LightGCN(n_params, args, norm_mat_p, norm_mat_c, norm_mat_v).to(device)

    ## 定义优化器
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # lr是learning rate的缩写，值为1e-4

    print("start training ...")
    best_recall = 0.0  # 最佳评估指标
    count = 0  # 计数器，用来记录best_recall未被更新的次数，本实验中为超过10次
    for epoch in range(args.epoch):  # 实验轮次，默认训练200轮
        print("epoch: ", epoch)
        f = open(fname, 'a')
        now = datetime.datetime.now()
        f.write(now.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write("epoch: " +  str(epoch) + '\n')
        f.close()
        # shuffle training data # 打乱训练数据的顺序
        train_p_ = train_p
        index = np.arange(len(train_p_))
        np.random.shuffle(index)
        train_p_ = train_p_[index].to(device)
        type_num_ = type_num.to(device)

        """training"""
        model.train()
        loss, s = 0, 0  # 训练损失和数据量
        hits = 0  # 未使用

        while s + args.batch_size <= len(train_p):  # 剩余数据还能训练一个批次，一个批次大小batch_size=4096
            #print("batch: ", c)
            if s + 2 * args.batch_size <= len(train_p):  # 剩余数据至少有两个批次
                batch = get_feed_dict(train_p_,
                                      type_num_,
                                      user_dict['train_user_set_p'],
                                      s, s + args.batch_size,
                                      n_negs)

                batch_loss = model(batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                s += args.batch_size
            else:                    # 剩余数据不足一个批次，即:s + args.batch_size <= len(train_p) < s + 2 * args.batch_size
                batch = get_feed_dict(train_p_,
                                      type_num_,
                                      user_dict['train_user_set_p'],
                                      s, s + args.batch_size,
                                      n_negs)
                batch_loss = model(batch)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                s += args.batch_size

                batch = get_feed_dict(train_p_,
                                      type_num_,
                                      user_dict['train_user_set_p'],
                                      s, len(train_p),
                                      n_negs)

                batch_loss = model(batch)   # 计算单批损失

                optimizer.zero_grad()  # 梯度归零
                batch_loss.backward()  # 反向传播，计算得到每个参数的梯度值
                optimizer.step()       # 根据梯度下降更新网络参数

                loss += batch_loss     # 计算每一批损失的总和
                s += args.batch_size   # 每次处理一个批大小的数据


        #把运行结果写入文件
        f = open(fname, 'a')
        f.write("loss: " + str(loss) + '\n')
        f.close()
        # K = [10, 40, 80]
        # 10
        recall_tem, ndcg_tem = test(model, user_dict, n_params,
                                args.topK1)  # 测试模型得到两个评估指标recall & NDCG, test函数在evaluate.py文件里

        f = open(fname, 'a')
        now = datetime.datetime.now()
        f.write(now.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write(str(args.topK1) + ':' + "recall: " + str(recall_tem) + " ndcg: " + str(ndcg_tem) + '\n')

        # 40
        recall_tem, ndcg_tem = test(model, user_dict, n_params,
                                args.topK2)  # 测试模型得到两个评估指标recall & NDCG, test函数在evaluate.py文件里
        # if recall_tem > best_recall:
        #     best_recall = recall_tem
        #     count = 0
        # else:
        #     count += 1
        # if count > 30:  # best_recall未被更新的次数超过10次
        #     break
        f.write(str(args.topK2) + ':' + "recall: " + str(recall_tem) + " ndcg: " + str(ndcg_tem) + '\n')
        # 80
        recall_tem, ndcg_tem = test(model, user_dict, n_params,
                                args.topK3)  # 测试模型得到两个评估指标recall & NDCG, test函数在evaluate.py文件里
        f.write(str(args.topK3) + ':' + "recall: " + str(recall_tem) + " ndcg: " + str(ndcg_tem) + '\n')
        f.close()

    print("recall: ",best_recall,"ndcg: ",ndcg_tem )  # 打印最佳评估指标

if __name__ == '__main__':
    main()
