import os
import numpy as np
from sklearn.model_selection import train_test_split
import chess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt
import pandas as pd



# 准备标签
def prepare_labels(moves_list):
    all_labels = []
    for move_sequence in tqdm(moves_list, desc="Processing Moves Labels", unit="game"):
        board = chess.Board()
        move_sequence = move_sequence.split()
        game_labels = []
        for move in move_sequence:
            move_index = move_to_integer(move, board)
            game_labels.append(move_index if move_index != -1 else 0)  # 处理非法步
            board.push(chess.Move.from_uci(move))
        all_labels.append(game_labels)
    return all_labels

def move_to_integer(move, board):
    move_obj = chess.Move.from_uci(move)
    legal_moves = list(board.legal_moves)
    try:
        return legal_moves.index(move_obj)
    except ValueError:
        return -1  # 非法棋步处理


# 用于数据集的包装
class ChessDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# CNN 模型定义
class ChessCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChessCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),  # 全局池化到 1x1
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 转换形状为 [batch_size, channels, height, width]
        return self.model(x)


# 使用内存映射加载数据
def load_batch_with_mmap(file_path):
    return np.load(file_path, mmap_mode='r')  # 'r' 模式表示只读方式加载

# 处理批次数据并训练模型
def process_batches_in_stream(output_dir, model, criterion, optimizer, epochs=20, batch_size=512, test_size=0.2):
    df = pd.read_csv('chess_games.csv')
    moves_list = df['Moves']
    all_labels = prepare_labels(moves_list)
    batch_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    batch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for batch_file in batch_files:
        file_path = os.path.join(output_dir, batch_file)
        batch_data = load_batch_with_mmap(file_path)
        print(f"Processing {batch_file} with shape: {batch_data.shape}\n")
        
        cnn_input_flattened, labels_flattened = [], []
        for game_data, game_labels in tqdm(zip(batch_data, all_labels), total=len(batch_data), desc="Processing"):
            steps_data, steps_labels = game_data.shape[0], len(game_labels)
            min_steps = min(steps_data, steps_labels)
            
            cnn_input_flattened.extend(game_data[:min_steps])
            labels_flattened.extend(game_labels[:min_steps])

        cnn_input_flattened = np.array(cnn_input_flattened)
        labels_flattened = np.array(labels_flattened)

        # 重新 reshape 为 (144000, 8, 8, 12)
        cnn_input_reshaped = cnn_input_flattened.reshape(-1, 8, 8, 12)
        print(f"cnn_input_reshaped shape: {cnn_input_reshaped.shape}\n")

        X_train, X_test, y_train, y_test = train_test_split(
            cnn_input_reshaped, labels_flattened, 
            test_size=test_size, random_state=42
        )

        train_dataset = ChessDataset(X_train, y_train)
        test_dataset = ChessDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 训练过程
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss):
                    print("NaN loss detected, skipping this batch")
                    continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        # 测试模型 (可选)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), 'chess_cnn_model.pth')


# 特征图展示
def visualize_feature_maps(model, input_data):
    model.eval()
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor = input_tensor.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]

    with torch.no_grad():
        feature_maps = []
        x = input_tensor
        for layer in model.model:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                feature_maps.append(x)
            else:
                x = layer(x)

    for i, feature_map in enumerate(feature_maps, start=1):
        num_features = feature_map.shape[1]
        rows = 4  # Set the number of rows to make the visualization more rectangular
        cols = (num_features + rows - 1) // rows  # Calculate the number of columns based on rows
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        fig.suptitle(f'Step {i} - Layer {i} Feature Maps', fontsize=16)
        feature_map = feature_map.squeeze(0).cpu().numpy()
        for j in range(num_features):
            row, col = divmod(j, cols)
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.imshow(feature_map[j], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Feature {j+1}')
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()



def main():
    
    model = ChessCNN(num_classes=80).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 假设存储批次数据的目录
    output_directory = "cnn_batches"

    # 调用处理函数，逐批处理数据
    process_batches_in_stream(output_directory, model, criterion, optimizer)

    # 发送数据到模型中，并反馈特征图
    # 假设 input_data 为此处的一个数据标签
    input_data_example = np.random.rand(8, 8, 12)  # Example input data for visualization
    visualize_feature_maps(model, input_data_example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    main()
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
