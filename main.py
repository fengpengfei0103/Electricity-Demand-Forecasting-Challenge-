import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# 数据预处理
def preprocess_data(train_data, test_data):
    # 处理id列
    label_encoder = LabelEncoder()
    train_data['id'] = label_encoder.fit_transform(train_data['id'])
    test_data['id'] = label_encoder.transform(test_data['id'])

    # # 归一化dt列
    # scaler = MinMaxScaler()
    # train_data['dt'] = scaler.fit_transform(train_data[['dt']])
    # test_data['dt'] = scaler.transform(test_data[['dt']])
    #
    # # 处理type列
    # train_data['type'] = label_encoder.fit_transform(train_data['type'])
    # test_data['type'] = label_encoder.transform(test_data['type'])

    return train_data, test_data


train_data, test_data = preprocess_data(train_data, test_data)


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, target=None):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.target is not None:
            return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.target[idx],
                                                                                   dtype=torch.float32)
        else:
            return torch.tensor(self.data[idx], dtype=torch.float32)


# 构建数据集和数据加载器
X_train = train_data[['id', 'dt', 'type']].values
y_train = train_data['target'].values
X_test = test_data[['id', 'dt', 'type']].values

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

test_dataset = CustomDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 定义模型
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMWithAttention, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加时间步维度
        h_lstm, _ = self.bilstm(x)
        h_att, _ = self.attention(h_lstm, h_lstm, h_lstm)
        h_att = h_att.mean(dim=1)
        out = self.fc(h_att)
        return out

# 获取词汇表大小
print(X_train.shape)
print(X_train)
print(X_test.shape)
print(X_test)
input_dim = 3
print(input_dim)
# 模型实例化
model = BiLSTMWithAttention(input_dim=3, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    num = 0
    for inputs, targets in train_loader:
        print(num)
        num = num + 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 进行预测
model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().tolist())

# 输出预测结果
test_data['target'] = predictions
test_data.to_csv('test_predictions.csv', index=False)
