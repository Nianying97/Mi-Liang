import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
import numpy as np
from torch import nn, optim
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class GlobalGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.context = nn.Parameter(torch.randn(1, out_dim))  # 全局图上下文
        self.attn_proj = nn.Linear(2 * out_dim, 1)

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.size()
        h = self.W(x)  # [B, N, D']
        context = self.context.expand(B, -1)  # [B, D']
        context = context.unsqueeze(1).expand(-1, N, -1)  # [B, N, D']

        concat = torch.cat([h, context], dim=-1)  # [B, N, 2D']
        scores = self.attn_proj(concat).squeeze(-1)  # [B, N]
        attn_weights = torch.softmax(scores, dim=-1)  # [B, N]
        attn_weights = attn_weights.unsqueeze(-1)  # [B, N, 1]
        out = x * attn_weights  # [B, N, D]
        return out  # 注意力加权后的节点特征


class TemporalCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        # x = self.conv1(x)
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))  # 少一个卷积层，其威力大减
        x = self.pool(x).squeeze(-1)  # [B, out_dim]
        return x


class TV_Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.temporal_encoder = TemporalCNN(input_dim*16, hidden_dim=128, out_dim=128)
        self.variable_attention = GlobalGraphAttention(input_dim, out_dim=input_dim)

        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):  # x: [B, T, N, D]
        B, T, N, D = x.shape
        attn_feats = []

        for t in range(T):
            xt = x[:, t, :, :]  # [B, N, D]
            out_t = self.variable_attention(xt)  # [B, N, D]
            attn_feats.append(out_t)

        v_feat = torch.stack(attn_feats, dim=1)  # [B, T, N, D]
        v_feat = v_feat.view(B, T, N * D)        # [B, T, N*D]

        t_feat = self.temporal_encoder(v_feat)   # [B, 128]
        out = self.classifier(t_feat)            # [B, num_classes]
        return out

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

model = TV_Net(input_dim=26, num_classes=7).to(device)
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