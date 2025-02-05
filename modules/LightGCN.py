import math

import numpy as np
import torch
import torch.nn as nn        # pytorch自带的函数库，包含了神经网络中使用的一些常用函数
from utils.contrast import Contrast
class GraphConv(nn.Module):  # 自定义层/模型，通过继承nn.Module实现
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.2, mess_dropout_rate=0.1):  # 申明各个层的参数定义
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops                        # n_hops = 3
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate  # edge_dropout_rate=mess_dropout_rate=0.5

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):  # 对数据进行正则化处理
        noise_shape = x._nnz()  # 统计矩阵有多少个0

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)  # 返回[0,1]均匀分布的张量，noise_shape为张量形状
        # 构建张量的过滤器（掩码）<torch.floor返回具有random_tensor元素下限的新张量，torch.bool是张量的数据类型>
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices() # 返回一个序号网格数组<indices()用于提取数组元素或对数组进行切片>
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))   # 返回正则化后的稀疏矩阵


    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):  # 在forward中实现各个层的连接关系，即向前传播

        all_embed = torch.cat([user_embed, item_embed], dim=0)  # 对用户和项目张量进行按行(维数dim=0)拼接
        agg_embed = all_embed
        embs = [all_embed]

        # user_embed: 21 * 256, item_embed: 501 * 256, all_embed: 522 * 256

        for hop in range(self.n_hops):  # 循环3次
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)       # 2维 ==> 3维
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]  # [21716, 4, 256], [7977, 4, 256]

class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat_p, adj_mat_c, adj_mat_v):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat_p = adj_mat_p
        self.adj_mat_c = adj_mat_c
        self.adj_mat_v = adj_mat_v

        self.decay = args_config.l2 #1e-4
        self.emb_size = args_config.dim #256
        self.context_hops = args_config.context_hops #3
        self.mess_dropout = args_config.mess_dropout #False
        self.mess_dropout_rate = args_config.mess_dropout_rate #0.5
        self.edge_dropout = args_config.edge_dropout #False
        self.edge_dropout_rate = args_config.edge_dropout_rate #0.5
        self.pool = args_config.pool #'mean'
        self.n_negs = args_config.n_negs #64
        self.ns = args_config.ns #'mixgcf'
        self.K = args_config.K #5
        self.tua0 = args_config.tua0 #1.0
        self.tua1 = args_config.tua1 #0.5
        self.tua2 = args_config.tua2 #0.2
        self.tua3 = args_config.tua3 #0.2
        self.lamda = args_config.lamda #0.2
        self.mu = args_config.mu  # 0.1


        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        # 用普通的GCN初始化embedding (刚开始的embedding是无规律的需要送到模型训练)
        self.gcn_p = self._init_model_p()
        self.gcn_c = self._init_model_c()
        self.gcn_v = self._init_model_v()

        self.lear1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.lear2 = nn.Linear(self.emb_size, self.emb_size)

        self.lear3 = nn.Linear(self.emb_size * 3, self.emb_size)
        self.lear4 = nn.Linear(self.emb_size, self.emb_size)
        self.w = torch.nn.Parameter(torch.FloatTensor([0.4,0.3,0.3]), requires_grad=True)
        self.weightu = torch.nn.Parameter(torch.FloatTensor(self.emb_size, self.emb_size))
        self.weighti = torch.nn.Parameter(torch.FloatTensor(self.emb_size*3, self.emb_size))
        self.weightin = torch.nn.Parameter(torch.FloatTensor(self.emb_size * 3, self.emb_size))
        torch.nn.init.uniform_(self.weightu, 0, 1)
        torch.nn.init.uniform_(self.weighti, 0, 1)
        torch.nn.init.uniform_(self.weightin, 0, 1)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]  3种行为的邻接矩阵
        self.sparse_norm_adj_p = self._convert_sp_mat_to_sp_tensor(self.adj_mat_p).to(self.device)
        self.sparse_norm_adj_c = self._convert_sp_mat_to_sp_tensor(self.adj_mat_c).to(self.device)
        self.sparse_norm_adj_v = self._convert_sp_mat_to_sp_tensor(self.adj_mat_v).to(self.device)

    def _init_model_p(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj_p,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)  # 返回的是普通的GCN

    def _init_model_c(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj_c,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _init_model_v(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj_v,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col]) # un+in * un+in
        v = torch.from_numpy(coo.data).float() # data
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def calculate_theta_norm_2(self):
        # 对用户和物品嵌入矩阵应用L2正则化
        user_embed_norm = torch.norm(self.user_embed, p=2).pow(2)
        item_embed_norm = torch.norm(self.item_embed, p=2).pow(2)

        # 对其他可训练参数应用L2正则化
        weightu_norm = torch.norm(self.weightu, p=2).pow(2)
        weighti_norm = torch.norm(self.weighti, p=2).pow(2)
        weightin_norm = torch.norm(self.weightin, p=2).pow(2)

        # 将所有正则化项相加
        theta_norm_2 = user_embed_norm + item_embed_norm + weightu_norm + weighti_norm + weightin_norm
        return theta_norm_2

    def forward(self, batch=None):
        user = batch['users']
        type_num = batch['type_n']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        # print("aaaaaa: ",self.emb_size)

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        # pachas_uall, pachas_iall size: [21716, 4, 256]
        pachas_uall, pachas_iall = self.gcn_p(self.user_embed,
                                              self.item_embed, # 最初的两个embedding
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        cart_uall, cart_iall = self.gcn_c(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        view_uall, view_iall = self.gcn_v(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        batch_size = len(user)

        pachas_u = pachas_uall[user]   # pachas_u = [4096, 4, 256]
        pachas_i_pos = pachas_iall[pos_item]    # pachas_i_pos = [4096, 4, 256]
        #print("pachas_i_pos的维度是：",pachas_i_pos.size())
        pachas_i_neg = pachas_iall[neg_item[:, :self.K]]    # pachas_i_neg = [4096, 5, 4, 256]
        #print("pachas_i_neg的维度是：", pachas_i_neg.size())

        cart_u = cart_uall[user]
        cart_i_pos = cart_iall[pos_item]
        cart_i_neg = cart_iall[neg_item[:, :self.K]]

        view_u = view_uall[user]
        view_i_pos = view_iall[pos_item]
        view_i_neg = view_iall[neg_item[:, :self.K]]
        # print(type_num)

        #u_emb = torch.cat((pachas_u, cart_u, view_u), 2)
        #print(u_emb.size())
       # print(type_num[0].unsqueeze(dim=1))
       # print(pachas_u.squeeze(dim=0))
        # 对应论文里的公式4
        w0=type_num[0].unsqueeze(dim=1).unsqueeze(dim=1)    #[4096, 1, 1]
        #print("w0的形状：", w0.size())
        w1=type_num[1].unsqueeze(dim=1).unsqueeze(dim=1)
        w2=type_num[2].unsqueeze(dim=1).unsqueeze(dim=1)  # type_num里的元素值，单个用户不同行为下的占比，作为权重

        w_0 = torch.exp(w0) / (torch.exp(w0)+torch.exp(w1)+torch.exp(w2))  # 对这种权重进行了处理 [4096, 1, 1]
        #print("w_0的形状：", w_0.size())
        w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))

        # w_0 = type_num[0].unsqueeze(dim=1).unsqueeze(dim=1)
        # w_1 = type_num[1].unsqueeze(dim=1).unsqueeze(dim=1)
        # w_2 = type_num[2].unsqueeze(dim=1).unsqueeze(dim=1)

        # [4096, 1, 1]，[4096, 4, 256]
        u_emb = w_0.mul(pachas_u) + w_1.mul(cart_u) + w_2.mul(view_u)  # 论文里的公式5
        #print(u_emb.size())
        # 论文里的公式6
        pos_i_emb = torch.cat((pachas_i_pos, cart_i_pos, view_i_pos), 2)
        neg_i_emb = torch.cat((pachas_i_neg, cart_i_neg, view_i_neg), 3)

        #BPR_loss = self.create_bpr_loss(u_emb, pos_i_emb, neg_i_emb)
        # 修改之后的BPR_loss：不可行，维度不匹配
        #BPR_loss = self.create_bpr_loss(u_emb, pachas_i_pos, pachas_i_neg)

        # 修改之后的BPR_loss：不可行，维度不匹配
        #BPR_loss = self.create_bpr_loss(pachas_u, pachas_i_pos, pachas_i_neg)

        pur_i_pos_emb = torch.cat((pachas_i_pos, pachas_i_pos, pachas_i_pos), 2)
        pur_i_neg_emb = torch.cat((pachas_i_neg, pachas_i_neg, pachas_i_neg), 3)
        # 修改之后的BPR_loss
        BPR_loss = self.create_bpr_loss(u_emb, pur_i_pos_emb, pur_i_neg_emb)

        # 求购买的评分
        pur_u_i_emb = pur_i_neg_emb

        #infoNCE loss
        u_p = self.pooling(pachas_u)  # [4096, 256]
        u_c = self.pooling(cart_u)
        u_v = self.pooling(view_u)

        i_p = self.pooling(pachas_i_pos)
        i_c = self.pooling(cart_i_pos)
        i_v = self.pooling(view_i_pos)

        adj_u = torch.eye(batch_size).to(self.device)
        adj_i = torch.eye(batch_size).to(self.device)

        #self.tua0，1，2
        contr0 = Contrast(self.tua0) # self.tua0 = 1.0
        contr1 = Contrast(self.tua1) # self.tua1 = 0.5
        contr2 = Contrast(self.tua2) # self.tua2 = 0.2
        contr3 = Contrast(self.tua3) # self.tua3 = 0.2

        l0 = contr0.forward(u_p, u_v, adj_u)
        l1 = contr1.forward(u_p, u_c, adj_u)
        l2 = contr2.forward(i_p, i_c,adj_i)
        l3 = contr3.forward(i_p, i_v, adj_i)

        info_NCE = l0 + l1 +l2 +l3   # 论文里的公式12
        if torch.isnan(BPR_loss).any().item():
            print()
            print()
        if torch.isnan(info_NCE).any().item():
            print()
            print()
        # 添加theta_norm_2计算
        theta_norm_2 = self.calculate_theta_norm_2()

        #return BPR_loss + self.lamda * info_NCE
        # 修改后的公式13
        return BPR_loss + self.lamda * info_NCE + self.mu * theta_norm_2


    ## 池化层（用来降维）[4096, 4, 256] ==> [4096, 256]
    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':  # 执行这一部分
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self): # 未被调用
        pachas_uall, pachas_iall = self.gcn_p(self.user_embed,  # 购买行为下用户和物品的embedding
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        cart_uall, cart_iall = self.gcn_c(self.user_embed,
                                          self.item_embed,
                                          edge_dropout=self.edge_dropout,
                                          mess_dropout=self.mess_dropout)
        view_uall, view_iall = self.gcn_v(self.user_embed,
                                          self.item_embed,
                                          edge_dropout=self.edge_dropout,
                                          mess_dropout=self.mess_dropout)

        user_purchase, item_purchase = self.pooling(pachas_uall), self.pooling(pachas_iall)
        user_cart, item_cart = self.pooling(cart_uall), self.pooling(cart_iall)
        user_view, item_view = self.pooling(view_uall), self.pooling(view_iall)
        user_gcn_emb = torch.cat((user_purchase, user_cart, user_view), 1)
        item_gcn_emb = torch.cat((item_purchase, item_cart, item_view), 1)
        return user_gcn_emb, item_gcn_emb

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]
        # print("user_gcn_emb.size()是：",user_gcn_emb.size())
        # user_gcn_emb = [4096, 4, 256]
        # self.pooling(user_gcn_emb) = [4096, 256]
        # self.weightu = [256, 256]
        # u_e = [4096, 256]
        u_e = self.pooling(user_gcn_emb).mm(self.weightu)
        # print("weightu.size()是：", self.weightu.size())
        # print("u_e.size是：",u_e.size())
        # print("pos_gcn_embs.size是：", pos_gcn_embs.size())
        # pos_gcn_embs = [4096, 4, 768]
        # self.pooling(pos_gcn_embs) = [4096, 768]
        # self.weighti = [768, 256]
        # pos_e = [4096, 256]
        pos_e = self.pooling(pos_gcn_embs).mm(self.weighti)
        # print("pos_e.size()是：",pos_e.size())
        # print("weighti.size()是：", self.weighti.size())
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1).matmul(self.weightin)

        posonevactor = torch.Tensor([1] * len(u_e)).to(self.device)
        negonevactor = torch.Tensor([[1 for col in range(5)] for row in range(len(u_e))]).to(self.device)
        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)   # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size
        #print(mf_loss)
        return mf_loss + emb_loss