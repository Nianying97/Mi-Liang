import itertools
from collections import Counter

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
import numpy as np
from torch import nn, optim
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GaussianPriorAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可训练高斯权重控制参数

    def forward(self, x):
        B, T, D = x.shape
        # 高斯先验邻接矩阵 [T, T]
        pos = torch.arange(T, device=x.device).float().unsqueeze(1)
        dist = (pos - pos.T).pow(2)  # pairwise squared distance
        gaussian_weight = torch.exp(-dist / (2 * self.alpha ** 2))  # [T, T]

        # 引导注意力
        attn_out, _ = self.attn(x, x, x, attn_mask=None)
        # attn_out: [B, T, D], gaussian_weight: [T, T]
        out = torch.matmul(gaussian_weight, attn_out)  # → [B, T, D]
        return out  # 广播乘权


class CTNet(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_classes, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.gaussian_transformer = GaussianPriorAttention(model_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

        # 重建辅助模块（可选）
        self.reconstruct_layer = nn.Linear(model_dim, input_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, D, 1]
            nn.Flatten(),             # [B, D]
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes))

    def forward(self, x, mask=None):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x = x.view(B, T, N * F)  # 合并节点和特征维度

        x = self.input_proj(x)  # [B, T, D]
        x = self.pos_encoder(x)  # 位置编码
        x = self.gaussian_transformer(x)  # 高斯先验 transformer
        x = self.dropout(x)

        # 主任务：分类
        cls_out = self.classifier(x.transpose(1, 2))  # [B, D, T] → [B, num_classes]

        # 辅助任务：重建关键时间戳  若不使用 mask，仅输出分类预测 [B, num_classes]
        if mask is not None:
            masked_x = x[mask]  # 被 mask 的位置输出
            recon_out = self.reconstruct_layer(masked_x)  # [M, F]
            return cls_out, recon_out  # 返回分类与重建
        else:
            return cls_out

# 1. 加载数据
seq_features = np.load("../my-model/spt_seq_features3-1.npy")
seq_labels = np.load("../my-model/spt_seq_labels3-1.npy")
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

model = CTNet(input_dim=416, model_dim=256, num_heads=8, num_classes=7).to(device)
num_classes = 7

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

def get_cls_num_list(train_loader, num_classes):
    cls_count = Counter()
    for _, labels in train_loader:
        labels = labels.cpu().numpy()
        cls_count.update(labels.tolist())
    cls_num_list = [cls_count[i] if i in cls_count else 0 for i in range(num_classes)]
    return cls_num_list
cls_num_list = get_cls_num_list(train_loader, num_classes)
print("每类样本数：", cls_num_list)

# criterion = nn.CrossEntropyLoss()
criterion = LDAMFocalLoss(cls_num_list=cls_num_list, max_m=0.7, s=10, gamma=1)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))


#   训练集
train_losses = []
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        for batch_idx, (inputs, label) in enumerate(train_loader):

            optimizer.zero_grad()
            inputs, label = inputs.to(device), label.to(device)
            outputs = model(inputs)
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}')

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

    with torch.no_grad():                   # 梯度清零
        for inputs, label in test_loader:
            inputs, label = inputs.to(device), label.to(device)
            outputs = model(inputs)         # 实际输出
            label = label.long()        # 真实标签值
            _, predicted = torch.max(outputs.data, 1)       # 找出预测最高的类别索引

            total += label.size(0)                         # 所有的测试样本总量
            test_loss += criterion(outputs, label).item()  # 因为是 += ，所以要加 .item() 进行累加
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(label.cpu().numpy())
            for lab, pred in zip(label, predicted):
                class_count[lab.item()] += 1
                if lab == pred:
                    class_correct[lab.item()] += 1
    class_acc = class_correct/class_count

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