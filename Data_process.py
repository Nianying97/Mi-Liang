import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# # 设置文件夹路径
# folder_path = '../tcn-data-op'  # 例如：'./csv_files'
#
# # 创建一个空列表用于保存每个CSV文件的数据
# dataframes = []
#
# # 遍历文件夹中所有CSV文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_csv(file_path)  # 如果编码不是utf-8，可以加 encoding='gbk' 或其他
#         dataframes.append(df)
#
# # 将所有DataFrame按行拼接
# merged_df = pd.concat(dataframes, ignore_index=True)
# # 保存为一个新的CSV文件
# merged_df.to_csv('merged_data.csv', index=False)



# 数据预处理和划分时间序列数据
# 加载数据
df = pd.read_csv('../tcn-data-spt/SPT1.csv')  # CSV 文件路径

# 读取 CSV 文件，跳过空白行
# df = pd.read_csv("../tcn-data-spt/SPT3-1.csv", skip_blank_lines=True)
# # 去除全是空值的行（如果有）
# df = df.dropna(how='all')
# # 保存为新文件或覆盖原文件
# df.to_csv("../tcn-data-spt/SPT3-1.csv", index=False)

# 提取特征列和类别列    320个特征
features = df.iloc[:, 0:416]  # 仿真数据集 320， 实物实验416
label = df.iloc[:, 416]
# 对 label 进行数值编码
label_encoder = LabelEncoder()    # label转为数值型
df['label'] = label_encoder.fit_transform(df['label'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)
# 标准化特征列，每列均值 0；方差 1
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# 创建新的 Data ，包含标准化后的特征和原始的目标列
data = pd.DataFrame(features_scaled, columns=features.columns)
data['label'] = df['label']
# 假设特征列名为 feature1, feature2, ..., featureN，标签列名为 'label'  检查所有列是否都是数值型
all_numeric = data.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())

# 划分时间序列数据
feature_data = data.values[:, :-1]  # 去掉最后一列标签，保留特征列
labels = data.values[:, -1]         # 保存最后一列作为标签
n_node = 16  # 节点个数
origin_feats = 26   # 每个节点的特征维度
window_size = 7  # 每个窗口的时间步长
stride = 1        # 步长
# 生成窗口数据（假设 F = n_node * origin_feats）
windows = [feature_data[i:i+window_size, :] for i in range(0, feature_data.shape[0] - window_size + 1, stride)]
# 确保 windows 不是空列表
if not windows:
    raise ValueError("没有生成任何窗口，请检查数据维度和窗口参数！")
# 将 windows 转换为形状 [seq_len, n_node, origin_feats]  window_size = seq_len
reshaped_windows = []
for window in windows:
    # 将每个窗口的特征数据从 [window_size, F] 变换为 [window_size, n_node, origin_feats]
    reshaped_window = window.reshape(window_size, n_node, origin_feats)
    reshaped_windows.append(reshaped_window)
# 将列表转换为 numpy 数组，形状为 [seq_len, n_node, origin_feats]
seq_features = np.array(reshaped_windows)
seq_labels = labels[window_size-1::stride]    # 获取每个窗口的最后一个标签，保证数据和标签是配套的
np.save("spt_seq_features1.npy", seq_features)   # shape: [seq_len, nodes, features]
np.save("spt_seq_labels1.npy", seq_labels)    # shape: [seq_len, ]