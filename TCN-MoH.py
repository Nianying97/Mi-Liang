import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
import numpy as np
from torch import nn, optim
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

class TCNMoH(nn.Module):
    LOAD_BALANCING_LOSSES = []
    def __init__(
            self,
            dim,
            num_heads=8,
            proj_drop=0.,   # TCN的dropout
            shared_head=3,
            routed_head=3,  # 有两个不被选择  都是3头
            head_dim=None,  # 会自动计算
            tcn_layers=1,
            kernel_size=7,  # 时间卷积的层数和尺寸
            ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        if head_dim is None:  # dim = embed_dim升维后的
            self.head_dim = dim // num_heads    # 还是需要保证能够被整除
        else:
            self.head_dim = head_dim
        self.post_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        # 时间卷积
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2  # 确保序列长度不变
            # 创建卷积层
            conv_layer = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation)
            # 添加卷积层、激活函数和 dropout
            self.tcn_layers.append(conv_layer)
            self.tcn_layers.append(nn.ReLU())
            self.tcn_layers.append(nn.Dropout(proj_drop))

        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 定义最终输出的 dropout 层。
        self.shared_head = shared_head
        self.routed_head = routed_head   # num_heads - shared_head是总的路由器头数，routed_head是重要性前K的个数
        if self.routed_head > 0:  # 如果大于0，则进入这个条件块，表示需要为每个样本动态选择注意力头。
            # 通过这个门控机制，模型能够为每个样本选择不同的注意力头，从而提高模型在特定任务上的性能和灵活性。
            self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:  # 如果共享头数量大于0，则定义一个线性层用于路由头的权重计算。
                self.wg_0 = torch.nn.Linear(dim, 2, bias=False)
                # 示将输入特征映射到两个输出（可能用于指示选择哪个头）
        if self.shared_head > 1:  #  如果共享头数量大于1，则定义一个线性层用于计算共享头的权重。
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(416, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7))

    def forward(self, x):
        # 假设 x.shape = [B, T, N, D]
        B, T, NODE, D = x.shape
        # 转换为 [B, T, N*D]，即 [B, T, D']
        x = x.view(B, T, NODE * D)
        B, N, C = x.shape  # [B，T，D]
        x = x.transpose(1, 2)  # 将输入转换为 (B, C, N) 以匹配卷积的输入格式
        residual = x  # 保存输入作为残差
        for layer in self.tcn_layers:
            x = layer(x)  # 逐层应用每个操作     # 是只执行一次tcn_layers就是，即内部的多个卷积，不是要再循环tcn_layers
        x = residual + x    # 添加残差连接 shape（B,C,N）
        x = x.transpose(1, 2)
        x = self.post_layer_norm(x)  # 归一化应该是在dim维度上
        _x = x.reshape(B * N, C)  # MOH的输入应该为 B, N, C
        if self.routed_head > 0:   # 路由机制，决定那些头作为该样本的路由器头
            logits = self.wg(_x)   # 映射到 路由器头个数上来
            gates = F.softmax(logits, dim=1)  # 表示每个头的重要性分布
            # 将权重值转换为概率分布。gates 的每一行的和为 1，表示每个头的重要性分布。
            num_tokens, num_experts = gates.shape
            # num_tokens 是输入样本的数量（批量大小），num_experts 是可用的路由器头数量
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            # 动态路由器头的选择指的是每个样本根据自己的特征选择的路由器头对象可能不同，而不是说每个样本可以选择的路由器头的数量不同
            # 选择权重最高的 k 个头，这里 k 是 self.routed_head，表示要选择的路由器头数量。
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
            # 得到一个掩码 (mask)，在所有可用路由器数量中（不是最高K值），其中被选择的头的位置为 1，其余为 0。
            # 每个样本都会生成一个掩码，对应该样本所选择的 前K个路由器头
            # 暂定思想是选定前K个路由头，然后根据每个头与dim映射的权重，返回去找权重大的对应特征
            if self.training:  # 保证各个路由器头的负载是相对均衡的，避免某些头过度使用，而其他头几乎不用。
                me = gates.mean(dim=0)
                ce = mask.float().mean(dim=0)  # 计算门控值的平均值 me 和掩码的平均值 ce。
                l_aux = torch.mean(me * ce) * num_experts * num_experts
                # 门控机制还通过损失函数，动态调整特征的权重分配，以保证模型的负载均衡
                TCNMoH.LOAD_BALANCING_LOSSES.append(l_aux)  # 计算负载平衡损失 l_aux，并将其添加到类变量中
            routed_head_gates = gates * mask  # 这会把未被选中的头的权重置为 0，保证只有被选中的路由器头保留相应的权重。
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s  # 该样本中所有路由器头的权重，未被选中的都是0
            # 除以归一化分母 denom_s，使得每个样本中选中的路由器头的权重之和为 1。这样可以确保路由器头的权重经过了正确的归一化处理。
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
            # 将其变为 (B, N, num_heads) 的形状。
        if self.routed_head > 0:
            x = x.transpose(1, 2)
            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)  # 计算共享头的权重 shared_head_weight，并对其应用 softmax，生成共享头的门控值。
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
                # 应用 softmax 并重塑为 (B, N, 2)，表示对两个权重的处理。
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:  # 如果没有共享头
                masked_gates = routed_head_gates   # 如果没有共享头，则直接使用路由头的门控值
            x = x.transpose(1, 2)
            x = x.view(B, N, self.num_heads, self.head_dim)
            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
            # bne 与 bned 配对，以对 x 中的特征施加动态权重。
            # 这将根据 masked_gates 的权重来调整 x 中每个头的特征表现力
            # 这段代码的目的是根据共享头和路由头的权重动态调整输入特征的影响，
            # 增强模型在不同任务上的灵活性和性能。通过有效地结合共享头和路由头，模型能够更好地处理复杂特征和任务。
        else:  # 如果没有路由器头
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            x = x.transpose(1, 2)
            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
            # 路由器头（routed head）和共享头（shared head）在这段代码中确实用于处理单个样本之间特征权重的情况，从而影响最终的分类效果
            # 通过对每个样本的特征进行动态加权，模型能够更加关注那些在当前任务中重要的特征。
            # 这种方法可以提高模型的灵活性，使其在不同的任务或样本上表现得更好
        x = self.proj(x)
        x = self.proj_drop(x)
        x = torch.mean(x[:, :-1, :], dim=1)  # 平均池化  # [B, dim]
        out = self.classifier(x)  # [B, num_classes]
        return out  # shape: [B, N, dim]

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

model = TCNMoH(dim=416, num_heads=16, proj_drop=0., shared_head=6, routed_head=6, head_dim=None, tcn_layers=1, kernel_size=3).to(device)
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