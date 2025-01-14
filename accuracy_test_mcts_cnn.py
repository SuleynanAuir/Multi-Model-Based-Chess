import torch

# class ChessModelValidator:
#     def __init__(self, model, test_data, test_labels, batch_size=16):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = model.to(self.device)
#         self.test_data = test_data
#         self.test_labels = test_labels
#         self.batch_size = batch_size

#     def validate(self):
#         self.model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for i in range(0, len(self.test_data), self.batch_size):
#                 inputs = torch.tensor(self.test_data[i:i + self.batch_size], dtype=torch.float32).to(self.device)
#                 labels = torch.tensor(self.test_labels[i:i + self.batch_size], dtype=torch.long).to(self.device).view(-1)
#                 outputs = self.model(inputs).view(-1, 64)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         accuracy = correct / total
#         print(f"Test Accuracy: {accuracy * 100:.2f}%")

import numpy as np
import os
import torch
import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self, conv_size, conv_depth, num_classes):
        super(ChessModel, self).__init__()
        self.time_conv = nn.Conv2d(in_channels=12, out_channels=conv_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        layers = [nn.Conv2d(conv_size, conv_size, kernel_size=3, padding=1) for _ in range(conv_depth)]
        self.conv_layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_size * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        batch_size, steps, channels, height, width = x.shape
        x = x.view(batch_size * steps, channels, height, width)
        x = self.time_conv(x)
        x = self.relu(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.view(batch_size, steps, -1)
    
class ChessModelValidator:
    def __init__(self, model, data_dir, batch_size=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.data_dir = data_dir  # 传入数据目录
        self.batch_size = batch_size

        # 获取所有的 .npz 文件
        self.npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        # 随机选择一个 npz 文件
        npz_file = np.random.choice(self.npz_files)
        print(f"Selected file for testing: {npz_file}")
        
        # 加载数据
        data = np.load(os.path.join(data_dir, npz_file))
        self.test_data = data['boards']
        self.test_labels = data['labels']

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(self.test_data), self.batch_size):
                inputs = torch.tensor(self.test_data[i:i + self.batch_size], dtype=torch.float32).to(self.device)
                labels = torch.tensor(self.test_labels[i:i + self.batch_size], dtype=torch.long).to(self.device).view(-1)
                outputs = self.model(inputs).view(-1, 64)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    model = ChessModel(conv_size=32, conv_depth=3, num_classes=64)

# 加载模型的 state_dict
    checkpoint = torch.load('mcts_cnn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

# 将模型移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

# 创建验证器并验证
    validator = ChessModelValidator(model, data_dir='output_mcts_cnn')
    validator.validate()