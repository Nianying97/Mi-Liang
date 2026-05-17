import itertools
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

# 信息传播依赖的是 注意力加权聚合，不再需要静态的拉普拉斯平滑。
# 单个输入的 HGNN
# class DAHGNNConv(nn.Module):
#     def __init__(self, in_dims, out_dims, dropout=0.5, is_last=False):
#         super().__init__()
#         self.is_last = is_last
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.ReLU(inplace=True)
#         self.linear = nn.Linear(in_dims, out_dims, bias=False)
#         # 节点 → 超边 注意力参数
#         self.att_node2edge = nn.Parameter(torch.Tensor(1, 2 * out_dims))
#         # 超边 → 节点 注意力参数
#         self.att_edge2node = nn.Parameter(torch.Tensor(1, 2 * out_dims))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.linear.weight)
#         nn.init.xavier_uniform_(self.att_node2edge)
#         nn.init.xavier_uniform_(self.att_edge2node)
#
#     def forward(self, X: torch.Tensor, hg) -> torch.Tensor:
#         # 注意力机制是单头的
#         X = self.linear(X)  # 节点特征矩阵 [N, out_dims] 变化特征维度 可以不变化吗？
#         H = hg.incidence_matrix  # 关联矩阵 [N, M] binary matrix: node-edge incidence
#         # 节点到超边 Attention
#         N, M = H.shape
#         edge_features = torch.zeros((M, X.size(1)), device=X.device)
#         for j in range(M):
#             node_idx = torch.nonzero(H[:, j], as_tuple=False).squeeze()  # 节点索引 in edge j
#             x_i = X[node_idx]  # [k, out_dims]
#             edge_j = x_i.mean(dim=0, keepdim=True).repeat(len(x_i), 1)  # 超边中心向量
#             att_input = torch.cat([x_i, edge_j], dim=1)  # [k, 2*out_dims]
#             e = F.leaky_relu((att_input * self.att_node2edge).sum(dim=1))  # [k]
#             alpha = F.softmax(e, dim=0).unsqueeze(1)  # [k, 1]
#             edge_features[j] = torch.sum(alpha * x_i, dim=0)
#         # 超边到节点 Attention
#         new_X = torch.zeros_like(X)
#         for i in range(N):
#             edge_idx = torch.nonzero(H[i], as_tuple=False).squeeze()  # 与节点i相连的超边
#             e_j = edge_features[edge_idx]  # [k, out_dims]
#             node_i = X[i].unsqueeze(0).repeat(len(e_j), 1)  # [k, out_dims]
#             att_input = torch.cat([e_j, node_i], dim=1)  # [k, 2*out_dims]
#             e = F.leaky_relu((att_input * self.att_edge2node).sum(dim=1))  # [k]
#             beta = F.softmax(e, dim=0).unsqueeze(1)  # [k, 1]
#             new_X[i] = torch.sum(beta * e_j, dim=0)
#
#         # 是否使用 激活函数 和 dropout
#         if not self.is_last:
#             new_X = self.activation(new_X)
#             new_X = self.dropout(new_X)
#
#         return new_X


# 一个批次中同时处理

class DAHGNNConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, is_last=False, return_attention=False):
        super().__init__()
        self.is_last = is_last
        self.return_attention = return_attention
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.att_node2edge = nn.Parameter(torch.Tensor(1, 1, 1, 2 * out_dim))
        self.att_edge2node = nn.Parameter(torch.Tensor(1, 1, 1, 2 * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_node2edge)
        nn.init.xavier_uniform_(self.att_edge2node)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        X: [B, N, in_dim]
        new_X: [B, N, out_dim]
        H: [N, M] (static) or [B, N, M] (dynamic)
        """
        X = self.linear(X)  # [B, out_dim] 这一行是否可有可无 感觉作用不大  out_dim需要是nodes的整数倍
        B, _, _ = X.shape
        if H.dim() == 2:
            # Static H: expand to [B, N, M]  扩展一个维度
            H = H.unsqueeze(0).expand(B, -1, -1)

        B, N, _ = X.shape   # [B, N, out_dim]
        _, _, M = H.shape   # [B, N, M]

        # Step 1: Node → Edge 节点到超边聚合的注意力
        H_T = H.transpose(1, 2)  # [B, M, N]
        edge_mean = torch.bmm(H_T, X) / (H_T.sum(dim=2, keepdim=True) + 1e-6)  # [B, M, F_out]
        X_n2e = X.unsqueeze(2).repeat(1, 1, M, 1)  # [B, N, M, F_out]
        E_e = edge_mean.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, M, F_out]
        att_input = torch.cat([X_n2e, E_e], dim=-1)  # [B, N, M, 2*F_out]
        self.tau = 2.0  # init 中设置温度
        e = F.leaky_relu((att_input * self.att_node2edge).sum(dim=-1))  # [B, N, M]
        e = e.masked_fill(H == 0, float('-inf'))
        alpha = F.softmax(e / self.tau, dim=2).unsqueeze(-1)  # [B, N, M, 1]
        edge_feat = (alpha * X_n2e).sum(dim=1)  # [B, M, F_out]

        # Step 2: Edge → Node  超边到节点聚合的注意力
        E_e2n = edge_feat.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, M, F_out]
        X_exp = X.unsqueeze(2).repeat(1, 1, M, 1)  # [B, N, M, F_out]
        att_input = torch.cat([E_e2n, X_exp], dim=-1)  # [B, N, M, 2*F_out]
        e = F.leaky_relu((att_input * self.att_edge2node).sum(dim=-1))  # [B, N, M]
        e = e.masked_fill(H == 0, float('-inf'))
        beta = F.softmax(e, dim=2).unsqueeze(-1)  # [B, N, M, 1]
        new_X = (beta * E_e2n).sum(dim=2)  # [B, N, F_out]

        if not self.is_last:
            new_X = self.activation(new_X)
            new_X = self.dropout(new_X)

        if self.return_attention:
            return new_X, alpha, beta  # [B, N, M, 1] x 2
        else:
            return new_X


# 构建H3动态关联矩阵
# 计算节点间的余弦相似性，构建 top-k 关联矩阵
def build_hypermatrix(x, num_nodes, top_k):
    """ 输入:
        x: [B, N, d] 节点特征
        num_nodes: 节点数 N
        top_k: 每个节点取 top-k 相似节点
        输出:
        H: [B, N, N] 超图关联矩阵 """
    # 归一化特征
    normed = torch.nn.functional.normalize(x, dim=-1)  # [B, N, d]
    # 计算相似度矩阵：sim[b] = x[b] @ x[b].T
    sim = torch.bmm(normed, normed.transpose(1, 2))  # [B, N, N]
    # 对每个样本的每个节点，选取 top-k 相似节点的索引
    topk_idx = torch.topk(sim, top_k, dim=-1).indices  # [B, N, K]
    # 构建空超图关联矩阵
    H = torch.zeros_like(sim)  # [B, N, N]
    # 创建 batch 索引和节点索引
    B = x.size(0)
    device = x.device
    row_idx = torch.arange(B, device=device).view(-1, 1, 1)  # [B,1,1]
    col_idx = torch.arange(num_nodes, device=device).view(1, -1, 1)  # [1,N,1]
    # 赋值对应位置为1，表示某列（超边）中包含了哪些节点
    H[row_idx.expand_as(topk_idx), topk_idx, col_idx.expand_as(topk_idx)] = 1.0  # [B, N, N]
    return H  # [B, N, N]

# 基于成本敏感学习的Focal-loss损失函数，解决数据不平衡问题
# class CostSensitiveFocalLoss(nn.Module):
#     def __init__(self, num_classes, gamma=1.0, reduction='none'):
#         """  Cost-sensitive Focal Loss for multi-class classification.
#             Args:
#             alpha ：每个类别的权重（用于 cost-sensitive control）
#             gamma (float): 焦点参数，控制难分类样本的强调程度（Focal Loss的核心）.
#             reduction (str): 最终输出的聚合方式（mean、sum、或none）"""
#         super(CostSensitiveFocalLoss, self).__init__()
#
#         self.gamma = gamma
#         self.reduction = reduction
#         # 基于端到端的成本敏感学习 alpha参数，作为focal loss的类别权重 有助于模型分类
#         # self.alpha = nn.Parameter(torch.ones(num_classes), requires_grad=True)
#         # 初始alpha，默认1，不可训练，由外部传入
#         self.register_buffer('alpha', torch.ones(num_classes))
#
#     def set_alpha(self, alpha):
#         self.alpha = alpha.to(self.alpha.device)
#
#     def forward(self, logits, targets):
#         """ Args:
#             logits: 网络输出的未归一化得分，形状为 [B, C]
#             targets: 真实标签，形状为 [B]，每个值是类别索引 """
#         # 计算交叉熵损失（未求平均值）  reduction='none' 表示不对每个样本的损失进行聚合，保留 [B] 的损失值
#         ce_loss = F.cross_entropy(logits, targets, reduction='none')  # shape [B]
#         # 计算焦点词 pt 就是该样本真实label为3，softmax中预测为3类别的概率值为多少,此时还没有涉及focal-loss核心
#         # 因为 ce_loss = -log(pt),所以 pt = torch.exp(-ce_loss)
#         pt = torch.exp(-ce_loss)
#         # 当pt越接近1，说明预测的越准确，该类比较容易分类 focal_term → 0；反之pt越接近0，focal_term 趋近于 1
#         # gamma越大，越强调难样本
#         focal_term = (1 - pt) ** self.gamma
#         # 生成的 focal-loss运用到交叉熵损失函数上——基础focal损失
#         loss = focal_term * ce_loss   # focal-loss函数
#         # alpha_t = self.alpha.to(targets.device)[targets]
#         # loss = (alpha_t * loss)
#         alpha_t = self.alpha.to(targets.device)[targets]  # 按batch中每个样本对应类别索引取权重
#         loss = alpha_t * loss
#
#         reg = 0.01 * torch.norm(self.alpha, p=2)  # 可调节权重系数  加入正则项，防止 α 爆炸或塌缩
#         loss = loss + reg
#
#         if self.reduction == 'mean':  # 返回平均损失
#             return loss.mean()
#         elif self.reduction == 'sum':  # 返回总损失
#             return loss.sum()
#         else:
#             return loss  # 返回每个样本的 loss（用于进一步处理）

class LDAMFocalLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, gamma=1.0, weight=None):
        super().__init__()
        # 计算每类 margin：Delta_y
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float)))
        m_list = m_list * (max_m / torch.max(m_list))
        self.m_list = nn.Parameter(m_list, requires_grad=False)  # 不训练
        self.s = s
        self.gamma = gamma
        self.weight = weight  # 可选 class weights

    def forward(self, logits, target):
        device = logits.device
        target = target.long()
        # 构造 index mask
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), 1)

        # 生成 margin
        margin = torch.zeros_like(logits).to(device)
        m_list = self.m_list.to(device)  # 这一步很重要！
        margin[index] = m_list[target]
        logits_m = logits - margin  # 减去 margin
        logits_s = self.s * logits_m  # 缩放

        # CrossEntropy logits → prob
        log_probs = F.log_softmax(logits_s, dim=1)  # [B, C]
        probs = torch.exp(log_probs)  # softmax prob
        p_t = probs.gather(1, target.view(-1, 1)).squeeze(1)  # 正确类的 prob  计算 softmax 之后的概率
        # Focal 调整项
        focal_factor = (1 - p_t) ** self.gamma
        # 最终 loss
        CE_loss = F.nll_loss(log_probs, target, reduction='none', weight=self.weight)
        loss = focal_factor * CE_loss
        return loss.mean()


class WeightedGateFusion(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1))  # 每个模态一个权重分数)
    def forward(self, x1, x2, x3):
        # 每个分支计算注意力权重（标量）
        w1 = self.att_mlp(x1)  # shape: [batch_size, 1]
        w2 = self.att_mlp(x2)
        w3 = self.att_mlp(x3)
        # 对权重进行处理
        weights = torch.cat([w1, w2, w3], dim=-1)  # [batch_size, 3]
        weights = F.softmax(weights, dim=-1)  # 按模态归一化
        # w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]  # shape: [batch_size, 1]
        w1 = weights[..., 0:1]
        w2 = weights[..., 1:2]
        w3 = weights[..., 2:3]
        # 广播乘法进行融合
        fused = w1 * x1 + w2 * x2 + w3 * x3  # shape: [batch_size, out_dim]
        return fused


# 多模态多通道超图卷积
class MVHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k, num_nodes, dropout):
        super().__init__()
        self.branch = HGNNBranch(in_dim, hidden_dim, out_dim, dropout)
        self.gate = WeightedGateFusion(out_dim)
        self.k = k  # 5
        self.num_nodes = num_nodes  # 16

    def forward(self, X, H1, H2, return_attention=False):
        B, T, _, F = X.shape
        out_list = []
        attention_list = []

        for t in range(T):
            x_t = X[:, t, :, :]
            H3 = build_hypermatrix(x_t, self.num_nodes, self.k)

            if return_attention:
                out1, attn1, _ = self.branch(x_t, H1, return_attention=True)
                out2, attn2, _ = self.branch(x_t, H2, return_attention=True)
                out3, attn3, _ = self.branch(x_t, H3, return_attention=True)
            else:
                out1 = self.branch(x_t, H1)
                out2 = self.branch(x_t, H2)
                out3 = self.branch(x_t, H3)

            gated = self.gate(out1, out2, out3)
            out_list.append(gated)

            if return_attention:
                attention_list.append({
                    'alpha1': attn1[0], 'beta1': attn1[1],
                    'alpha2': attn2[0], 'beta2': attn2[1],
                    'alpha3': attn3[0], 'beta3': attn3[1],})

        output = torch.stack(out_list, dim=1)
        if return_attention:
            return output, attention_list
        else:
            return output

class HGNNBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = DAHGNNConv(in_dim, hidden_dim, dropout=dropout)
        self.conv2 = DAHGNNConv(hidden_dim, out_dim, is_last=True)
        # 第二层使用 is_last=True 来去掉激活 + dropout，使得输出更稳定用于分类或时间序列预测
    def forward(self, x, H, return_attention=False):
        if return_attention:
            self.conv1.return_attention = True
            self.conv2.return_attention = True
            x, alpha1, beta1 = self.conv1(x, H)
            x, alpha2, beta2 = self.conv2(x, H)
            return x, (alpha1, beta1), (alpha2, beta2)
        else:
            self.conv1.return_attention = False
            self.conv2.return_attention = False
            x = self.conv1(x, H)
            x = self.conv2(x, H)
            return x


# 在实测中再看是将所有节点的F拼接还是分开计算
class CNN1D(nn.Module):  # 提取突发时序特征
    def __init__(self, input_dim, out_dims):  # 这是单个节点的特征维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, out_dims, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))  # 输出 shape: [B*N, F, T] -> [B*N, F, 1]
    def forward(self, x):
        B, T, N, F = x.shape
        x = x.view(B, T, N * F)  # [B, T, N*F]
        x = x.transpose(1, 2)  # [B, N*F, T] —— CNN1D 的标准输入格式
        # CNN 在时间维度上滑动
        out = self.net(x)  # [B, F*N, 1]  只关注 每个节点自身随时间变化的特征序列，并没有考虑节点之间的结构关系
        out = out.squeeze(-1)  # [B, F]  去掉最后一个维度，去除最后一个时间步
        return out

# class LSTM(nn.Module):  # 提取周期性故障依赖性
#     def __init__(self, input_dim, out_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, out_dim, num_layers=2, batch_first=True)
#     def forward(self, x):
#         B, T, N, F = x.shape
#         x = x.view(B, T, N * F)  # [B, T, N*F]   也是每个节点的时序信息单独提取
#         out, _ = self.lstm(x)   # [B, T, F*N]
#         return out[:, -1, :]  # [B, F]  去掉中间T维度，去除最后一个时间步

# class AttentionLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.attn_layer = nn.Linear(hidden_dim * 2, 1)
#
#     def forward(self, x):
#         B, T, N, D = x.shape
#         x = x.view(B, T, N * D)  # [B, T, N*F]   也是每个节点的时序信息单独提取
#         lstm_out, _ = self.lstm(x)  # [B, T, 2*H]
#         attn_score = torch.softmax(self.attn_layer(lstm_out), dim=1)  # [B, T, 1]
#         out = torch.sum(attn_score * lstm_out, dim=1)  # [B, H]
#         return out

class ALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        #  全局上下文向量负责“判断标准”，注意力机制负责“执行聚焦”。它们是分工协作的。
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_layer = nn.Linear(hidden_dim * 4, 1)  # 注意力得分线性层（后续会用于计算时间步注意力）
        self.global_context = nn.Parameter(torch.randn(1, hidden_dim * 2))    # 全局上下文向量，学习得到（表示全局时间特征）
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # 降维

    def forward(self, x):  # x: [B, T, N, F]
        B, T, N, D = x.shape
        x = x.view(B, T, N * D)  # [B, T, N*F]
        lstm_out, _ = self.lstm(x)  # [B, T, 2H]
        # Step 3：生成全局时间上下文向量，并扩展为每个时间步使用
        global_ctx = self.global_context.expand(B, -1).unsqueeze(1)  # [B, 1, 2H]
        global_ctx = global_ctx.expand(-1, T, -1)  # [B, T, 2H]
        # Step 4：拼接 LSTM 输出和全局上下文，作为注意力输入
        attn_input = torch.cat([lstm_out, global_ctx], dim=-1)  # [B, T, 4H]
        # Step 5：计算注意力分数，并归一化
        attn_scores = self.attn_layer(attn_input)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]
        # Step 6：加权求和，聚合时间特征
        out = torch.sum(attn_weights * lstm_out, dim=1)  # [B, 2H]
        # out = self.down_proj(out)  # [B, H]
        return out


# class MemoryAugmentedLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, memory_slots=2, memory_dim=64):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.memory_slots = memory_slots  # 外部记忆块数目（表示“多少个周期模板槽位”）
#         self.memory_dim = memory_dim   # 每个记忆向量的维度
#         # LSTM 控制器
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
#         # 外部记忆：初始化为 learnable memory bank
#         self.memory = nn.Parameter(torch.randn(memory_slots, memory_dim))   # [memory_slots, memory_dim]
#         # 内容读取：attention查询
#         self.read_query_proj = nn.Linear(hidden_dim, memory_dim)   # 将 LSTM 的输出 [B, H] 映射到与 memory 对齐的维度 [B, D]，作为注意力“查询向量”
#         # 将当前隐藏状态 h_t 与 memory 读取结果 r_t 拼接后，通过线性变换变回 [B, H]，得到最终融合特征。
#         self.read_merge_proj = nn.Linear(hidden_dim + memory_dim, hidden_dim)
#
#     def forward(self, x):
#         B, T, N, D = x.shape
#         x = x.view(B, T, N * D)
#         batch_size, seq_len, _ = x.size()          # x: [B, T, F]
#         # LSTM 输出
#         lstm_out, _ = self.lstm(x)  # 得到每个时间步的隐藏状态，形状 [batch, seq_len, hidden_dim]
#         outputs = []
#         for t in range(seq_len):
#             h_t = lstm_out[:, t, :]  # 取得每个时间步的隐藏层状态  对每一个时间步做 memory attention
#
#             # 查询记忆结果
#             query = self.read_query_proj(h_t)  # [B, hidden_dim] -> [B, memory_dim] 查询向量
#             # 拓展 memory 到 batch 维度，变成 [batch, memory_slots, memory_dim]
#             mem = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [B, memory_slots, memory_dim] = 记忆memory
#             # 与 query 做点积注意力（计算每个 memory slot 与 query 的相似度），结果是 [B, memory_slots]，表示每个槽的注意力权重。
#             attn_score = F.softmax(torch.bmm(mem, query.unsqueeze(2)).squeeze(2), dim=1)  # [B, memory_slots]
#             # 以注意力权重对 memory 加权求和，得到 memory 读取结果 read_vec
#             read_vec = torch.bmm(attn_score.unsqueeze(1), mem).squeeze(1)  # [B, memory_dim]
#
#             # 读取结果和隐藏层状态融合
#             # 将记忆中的读取结果与原本的隐藏层状态进行拼接
#             merged = torch.cat([h_t, read_vec], dim=1)  # [B, Hidden_dim + memory_dim]
#             # 投影回隐藏空间作为最终表示。这里用 tanh 激活可以增强非线性建模能力
#             fused = torch.tanh(self.read_merge_proj(merged))  # [B, H]
#             # 将所有时间步融合结果拼接，得到整个序列的表示
#             outputs.append(fused.unsqueeze(1))  # [B, 1, H]
#         outputs = torch.cat(outputs, dim=1)  # [B, T, H]
#         out = outputs[:, -1, :]  # [B, H]  去掉中间T维度，去除最后一个时间步
#         return out

class GateFusion(nn.Module):
    def __init__(self, cnn_out, fused_dim):
        super().__init__()
        # 在加权融合前，必须将其映射到同一维度
        self.gate_fc = nn.Sequential(
            nn.Linear(cnn_out * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 1),
            nn.Sigmoid())
    def forward(self, cnn_feat, lstm_feat):
        fusion_input = torch.cat([cnn_feat, lstm_feat], dim=-1)  # [B, 2F]
        gate = self.gate_fc(fusion_input)                         # [B, 1]
        fused = gate * cnn_feat + (1 - gate) * lstm_feat   # [B, 2*F]       # [B, dim]
        return fused  # [B, F]

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_out, lstm_hidden, fused_dim, num_classes):
        super().__init__()
        self.cnn = CNN1D(input_dim, cnn_out)
        self.lstm = ALSTM(input_dim, lstm_hidden)
        self.gate_fusion = GateFusion(cnn_out, fused_dim)  # fused_dim只在求加权门控值时才会用到

        # 加权融合 各自已经进行了时序特征提取 所以时序维度变为1
        # 分类器有待改进
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x):
        cnn_feat = self.cnn(x)    # [B, N, F1] = [batch_size,nodes,cnn_out] -> [B, F]
        lstm_feat = self.lstm(x)  # [B, N, F2] = [batch_size,nodes,lstm_hidden] -> [B, F]
        fused = self.gate_fusion(cnn_feat, lstm_feat)  # [B, F]
        # x = fused.view(fused.size(0), -1)   # [B, F]  貌似无作用
        return self.classifier(fused)  # [B, num_classes]


class SP_MVHGNN(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden_dim, out_dim, k, num_nodes, dropout, cnn_out, lstm_hidden, fused_dim, num_classes):
        super().__init__()
        self.embedding_layer = nn.Linear(in_dim, embed_dim)
        self.mghgnn = MVHGNN(embed_dim, hidden_dim, out_dim, k, num_nodes, dropout)
        self.cnnlstm = CNNLSTM(out_dim*num_nodes, cnn_out, lstm_hidden, fused_dim, num_classes)

    def forward(self, X, H1, H2):
        # 1：特征增强 20 -> in_dim
        X = self.embedding_layer(X)
        # 2: 超图特征提取
        feature_seq = self.mghgnn(X, H1, H2)  # [B, T, N, embed_dim] -> [B, T, N, out_dim]
        # 3: 时序建模
        output = self.cnnlstm(feature_seq)  # cnnlstm 已经进行了分类
        return output

# H1 空间位置模态 将每个车厢内的节点作为一组超边
# 节点顺序 [ ADAU-BCU1-CCTV-CCU1-DCU1-ERM1-FDU-HMI1-HVAC-LIU-PAS1-PIS-SDAU1-TBU1-VCU-WTDU1]
H1 = [torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],], dtype=torch.float32)]
# 节点顺序 [ ADAU-BCU1-CCTV-CCU1-DCU1-ERM1-FDU-HMI1-HVAC-LIU-PAS1-PIS-SDAU1-TBU1-VCU-WTDU1]
H2 = [torch.tensor([
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],], dtype=torch.float32)]

# 1. 加载数据
seq_features = np.load("spt_seq_features3-1.npy")   # 仿真时用 seq_features6；seq_labels6
seq_labels = np.load("spt_seq_labels3-1.npy")    # 半实物时用 spt_seq_features3；spt_seq_labels3
# seq_features = np.load("seq_features6.npy")   # 仿真时用 seq_features6；seq_labels6
# seq_labels = np.load("seq_labels6.npy")    # 半实物时用 spt_seq_features3；spt_seq_labels3
# 2. 划分数据集
X = seq_features   # (batch_size, seq_len, n_nodes, num_features)
y = seq_labels     # (batch_size,)
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.3, random_state=42))
# 3. 初始化标准化器，并仅用训练集计算均值和标准差  均值为0，方差为1
scaler = StandardScaler()
num_features = seq_features.shape[2] * seq_features.shape[3]
X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
# 4. 训练集 转换为 PyTorch tensor格式
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)        # (batch_size, seq_len, n_nodes, num_features)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)           # (batch_size,)
# 5. 测试集 转换为 PyTorch tensor格式
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
# 6. 创建 DataLoader 加载数据  组装成数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# 7. 参数准备
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 7
# model = SP_MVHGNN(in_dim=20, embed_dim=32, hidden_dim=64, out_dim=32, k=5, num_nodes=16,
#                   dropout=0.5, cnn_out=256, lstm_hidden=128, fused_dim=256, num_classes=7).to(device)   # 仿真数据集参数

model = SP_MVHGNN(in_dim=26, embed_dim=26, hidden_dim=48, out_dim=32, k=5, num_nodes=16,
                  dropout=0.5, cnn_out=128, lstm_hidden=64, fused_dim=128, num_classes=7).to(device)

# model = SP_MMHGNN(in_dim=20, embed_dim=32, hidden_dim=32, out_dim=18, k=9, num_nodes=16,
#                   dropout=0.5, cnn_out=214, lstm_hidden=214, fused_dim=418, num_classes=7).to(device)

# 8. 损失函数  根据每类的数量自动分配权重，用作损失函数中的 alpha 参数
# criterion = CostSensitiveFocalLoss(num_classes=num_classes, gamma=1, reduction='mean')
# criterion = nn.CrossEntropyLoss()

def get_cls_num_list(train_loader, num_classes):
    cls_count = Counter()
    for _, labels in train_loader:
        labels = labels.cpu().numpy()
        cls_count.update(labels.tolist())
    cls_num_list = [cls_count[i] if i in cls_count else 0 for i in range(num_classes)]
    return cls_num_list
cls_num_list = get_cls_num_list(train_loader, num_classes)
print("每类样本数：", cls_num_list)

criterion = nn.CrossEntropyLoss()
# criterion = LDAMFocalLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, gamma=2)
# criterion = LDAMFocalLoss(cls_num_list=cls_num_list, max_m=0.7, s=10, gamma=1)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))


#   训练集
train_losses = []
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    losses = []
    accuracies = []
    # alpha_fixed = False
    # best_alpha = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)

        prev_bs = None
        H1_batch = None
        H2_batch = None
        for batch_idx, (inputs, label) in enumerate(train_loader):
            current_bs = inputs.size(0)
            if current_bs != prev_bs:
                H1_batch = H1[0].unsqueeze(0).repeat(current_bs, 1, 1).to(device)
                H2_batch = H2[0].unsqueeze(0).repeat(current_bs, 1, 1).to(device)
                prev_bs = current_bs

            optimizer.zero_grad()
            inputs, label = inputs.to(device), label.to(device)

            # ===== 打印 attention =====
            if batch_idx == len(train_loader) - 1:
                model.eval()
                with torch.no_grad():
                    # 获取 attention
                    feature_seq, attention_list = model.mghgnn(inputs, H1_batch, H2_batch, return_attention=True)
                    # 随机选一个样本（batch 内）
                    B = inputs.shape[0]
                    idx = torch.randint(0, B, (1,)).item()
                    print(f"\n[Epoch {epoch}] Final batch, randomly selected sample: {idx}")

                    attn = attention_list[0]  # 第一个时间步
                    for branch_name, alpha, beta in zip(
                            ['H1', 'H2', 'H3'],
                            [attn['alpha1'], attn['alpha2'], attn['alpha3']],
                            [attn['beta1'], attn['beta2'], attn['beta3']]):
                        alpha_np = alpha[idx, :, :, 0].cpu().numpy()  # shape [N, M]
                        beta_np = beta[idx, :, :, 0].cpu().numpy()  # shape [N, M]
                        print(f"\n--- {branch_name} ---")
                        print(f"Alpha (Node → Hyperedge) shape: {alpha_np.shape}")
                        print(alpha_np)
                        print(f"Beta (Hyperedge → Node) shape: {beta_np.shape}")
                        print(beta_np)
                model.train()

            outputs = model(inputs, H1_batch, H2_batch)
            loss = criterion(outputs, label.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            for i in range(num_classes):
                class_mask = (label == i)
                class_total[i] += class_mask.sum().item()
                class_correct[i] += ((predicted == i) & class_mask).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total != 0)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}')
        print("Per-class accuracy:", class_acc)
        # print("Current alpha_t:", criterion.alpha.data.cpu().numpy())

        # # 判断是否所有类别准确率都 > 0.99，满足则固定alpha
        # if (class_acc > 0.99).all() and not alpha_fixed:
        #     print(f"Epoch {epoch + 1}: All class accuracy > 0.99, fixing alpha.")
        #     best_alpha = criterion.alpha.clone()
        #     alpha_fixed = True
        #
        # # 如果未固定alpha，执行动态更新
        # if not alpha_fixed:
        #     new_alpha = criterion.alpha.cpu().clone()
        #     min_alpha = 0.1
        #     max_alpha = 2.0
        #     delta = 0.01
        #     for i in range(num_classes):
        #         if class_acc[i] < 0.7:
        #             new_alpha[i] = min(new_alpha[i] + delta, max_alpha)
        #         else:
        #             new_alpha[i] = max(new_alpha[i] - delta, min_alpha)
        #     new_alpha = torch.clamp(new_alpha, min=0.1, max=3.0)
        #     criterion.set_alpha(new_alpha.to(device))
        # else:
        #     # alpha固定，持续使用最佳alpha
        #     criterion.set_alpha(best_alpha.to(device))

num_epochs = 30
train_model(model, criterion, optimizer, train_loader, num_epochs=num_epochs)

#   测试集
num_classes = 7
classes = ['BS', 'BZC', 'IUC', 'MMP', 'MPBC', 'NO', 'QoS']
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    total = 0
    class_count = torch.zeros(num_classes)
    class_correct = torch.zeros(num_classes)
    prev_bs = None
    H1_batch = None
    H2_batch = None
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in test_loader:
            current_bs = inputs.size(0)
            if current_bs != prev_bs:
                H1_batch = H1[0].unsqueeze(0).repeat(current_bs, 1, 1).to(device)
                H2_batch = H2[0].unsqueeze(0).repeat(current_bs, 1, 1).to(device)
                prev_bs = current_bs

            inputs, label = inputs.to(device), label.to(device)
            outputs = model(inputs, H1_batch, H2_batch)         # 实际输出
            label = label.long()        # 真实标签值
            _, predicted = torch.max(outputs.data, 1)       # 找出预测最高的类别索引

            fea = outputs  # 应该是最后一个线性层的输出
            features.append(fea.cpu().numpy())
            labels.extend(label.cpu().numpy())  # 将获取的 features 和 labels 进行存储

            total += label.size(0)                         # 所有的测试样本总量
            test_loss += criterion(outputs, label).item()  # 因为是 += ，所以要加 .item() 进行累加
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(label.cpu().numpy())
            for lab, pred in zip(label, predicted):
                class_count[lab.item()] += 1
                if lab == pred:
                    class_correct[lab.item()] += 1
    class_acc = class_correct/class_count

    all_features = np.vstack(features)  # 将所有特征堆叠成一个大的数组
    all_labels = np.hstack(labels)  # 将标签转为一维数组
    # 保存 features 和 labels 到文件
    np.save('../t-SNE/tsne-mvhgnn-feature_adam.npy', all_features)
    np.save('../t-SNE/tsne-mvhgnn-label_adam.npy', all_labels)

    for i in range(num_classes):
        print(f'Accuracy of {classes[i]}: {class_acc[i]}')
    accu_score = accuracy_score(y_true, y_pred)          # 整体的准确度
    precision_sco = precision_score(y_true, y_pred, average='weighted')
    test_loss = test_loss / len(test_dataset)        # 整个的 损失loss  teat_dataset
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    normalized_cm = cm.astype('float')/class_count[:, np.newaxis]
    print(f'Accuracy on test on: {accu_score:.4f}')
    print(f'Precision on test on: {precision_sco:.4f}')
    print(f'Test set: Average loss: {test_loss:.4f}')
    print(f'F1 Score: {f1:.4f}, Recall: {recall:.4f}')
    def plot_confusion_matrix(cm, normalized_cm, classes, normalize=False, title='Confusion matrix'):
        """
        绘制混淆矩阵的函数。
        Parameters:
            cm (numpy.ndarray): 混淆矩阵
            classes (list): 类别标签列表
            normalize (bool, optional): 是否对混淆矩阵进行归一化，默认为 False
            title (str, optional): 图标题，默认为 'Confusion matrix'
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 创建子图，获取轴对象

        # 绘制非归一化混淆矩阵
        ax1 = axes[0]
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_title('Confusion matrix, without normalization')
        fig.colorbar(im1, ax=ax1)
        tick_marks = np.arange(len(classes))
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(classes, rotation=45)
        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(classes)
        fmt = 'd'  # 数值格式为整数
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax1.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        ax1.set_ylabel('True label')
        ax1.set_xlabel('Predicted label')

        # 绘制归一化混淆矩阵
        ax2 = axes[1]
        im2 = ax2.imshow(normalized_cm, interpolation='nearest', cmap='Blues')
        ax2.set_title('Normalized confusion matrix')
        fig.colorbar(im2, ax=ax2)
        ax2.set_xticks(tick_marks)
        ax2.set_xticklabels(classes, rotation=45)
        ax2.set_yticks(tick_marks)
        ax2.set_yticklabels(classes)
        fmt = '.3f' if normalize else 'd'  # 数值格式为浮点数或整数
        thresh = normalized_cm.max() / 2.
        for i, j in itertools.product(range(normalized_cm.shape[0]), range(normalized_cm.shape[1])):
            ax2.text(j, i, format(normalized_cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if normalized_cm[i, j] > thresh else "black")
        ax2.set_ylabel('True label')
        ax2.set_xlabel('Predicted label')
        plt.tight_layout()
    plot_confusion_matrix(cm, normalized_cm, classes, normalize=True, title='confusion matrix')
    plt.show()
test_model(model, test_loader)





