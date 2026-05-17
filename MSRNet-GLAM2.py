import itertools
import os
import time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
import numpy as np
from torch import nn, optim
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader


# ---------- Local Attention ----------
class LocalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.local_attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim),
            nn.Sigmoid())

    def forward(self, x):
        attn_weights = self.local_attn(x)
        return x * attn_weights  # 特征维度加权

# ---------- Global Attention ----------
class GlobalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.global_attn = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 单通道注意力分数
            nn.Sigmoid())

    def forward(self, x):
        scores = self.global_attn(x)  # [B, 1]
        return x * scores  # 全局统一加权

# ---------- Multi-Scale Residual Block ----------
class MultiScaleResBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim))
        self.branch2 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim))
        self.norm = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        out1 = self.branch1(x)
        # out2 = self.branch2(x)
        out = x + out1
        # 输入 out.shape = [B, T, F] = [128, 7, 256]
        out = out.transpose(1, 2)  # -> [B, F, T] = [128, 256, 7]
        out = self.norm(out)  # ✅ 正确使用 BatchNorm1d(256)
        out = out.transpose(1, 2)  # -> 恢复成 [B, T, F]
        return F.relu(out)

# ---------- MSRNet + GLAM ----------
class MSRNetGLAM(nn.Module):
    def __init__(self, input_dim=320, num_classes=7):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.msr_block1 = MultiScaleResBlock(128)
        self.msr_block2 = MultiScaleResBlock(128)

        self.local_attn = LocalAttention(128)
        self.global_attn = GlobalAttention(128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes))

    def forward(self, x):
        # 假设 x.shape = [B, T, N, F]
        B, T, N, D = x.shape
        # 转换为 [B, T, N*F]，即 [B, T, F']
        x = x.view(B, T, N * D)
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.msr_block1(x)
        # x = self.msr_block2(x)
        x = self.local_attn(x)
        x = self.global_attn(x)
        x = torch.mean(x, dim=1)  # [B, T, F] -> [B, F]
        return self.classifier(x)

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

model = MSRNetGLAM(input_dim=416, num_classes=7).to(device)
num_classes = 7

criterion = nn.CrossEntropyLoss()
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