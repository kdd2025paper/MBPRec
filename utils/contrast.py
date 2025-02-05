import torch
import torch.nn as nn
## 这部分与论文中的公式11有关
class Contrast(nn.Module):
    def __init__(self, tau):
        super(Contrast, self).__init__()
        self.tau = tau
    # # 返回相似度矩阵
    # def sim(self, z1, z2):   # z1,z2 = [4096, 256]
    #     z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    #     z2_norm = torch.norm(z2, dim=-1, keepdim=True)  # 计算z1和z2的二范数（每行求平方和，再开方）
    #     dot_numerator = torch.mm(z1, z2.t())                # 分子: z1与z2的转置相乘
    #     dot_denominator = torch.mm(z1_norm, z2_norm.t())    # 分母: z1的二范数向量(1列)与z2的二范数向量转置(1行)相乘==>[4096， 4096]
    #     sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)  # self.tau用来调参
    #     return sim_matrix  # sim_matrix = [4096, 4096] ==> 同一个用户在2个不同行为下的embedding距离
    # # 求某用户对比性损失函数值
    # def forward(self, z_mp, z_sc, pos):
    #     matrix_mp2sc = self.sim(z_mp, z_sc)  # 相似度矩阵
    #     matrix_sc2mp = matrix_mp2sc.t()      # 相似度矩阵的转置
    #
    #     matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8) # dim=1按行求和，qe-8防止分母为0
    #     lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()  # 先算log后算均值
    #
    #     matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
    #     lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()  # pos为对角线全为1的矩阵，后面的sum实际上求对角线上元素之和
    #     return (lori_mp + lori_sc)/2


    #修改对应论文中的公式11
    # 返回相似度矩阵
    def sim(self, z1, z2):   # z1,z2 = [4096, 256]
        dot_numerator = torch.mm(z1, z2.t())                # 分子: z1与z2的转置相乘
        sim_matrix = torch.exp(dot_numerator / self.tau)  # self.tau用来调参
        return sim_matrix  # sim_matrix = [4096, 4096] ==> 同一个用户在2个不同行为下的embedding距离

    def forward(self, z_mp, z_sc, pos):
        matrix_mp2sc = self.sim(z_mp, z_sc)  # 相似度矩阵
        matrix_sc2mp = matrix_mp2sc.t()      # 相似度矩阵的转置

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8) # dim=1按行求和，qe-8防止分母为0
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()  # 先算log后算均值

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()  # pos为对角线全为1的矩阵，后面的sum实际上求对角线上元素之和
        return lori_mp-lori_sc