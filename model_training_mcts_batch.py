import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
from data_preprocessing_batch_cnn_Mcts import ChessDataPreprocessor

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

class ChessDataset(Dataset):
    def __init__(self,npz_file):
        # 加载 npy文件，避免一次性加载所有数据到内存
        print(f"Loading data from {npz_file}")
        data = np.load(npz_file,mmap_mode = 'r')
        self.data = torch.tensor(data['boards'], dtype=torch.int8)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ChessTrainer:
    def __init__(self, model, data_dir, epochs=10, batch_size=16, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.npz_files_dir = data_dir
        self.npz_files = [f for f in os.listdir(self.npz_files_dir) if f.endswith('.npz')]  # List all npz files
        self.npz_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def load_batch_with_mmap(file_path):
        return np.load(file_path, mmap_mode='r')  # 'r' 模式表示只读方式加载

    def train(self):

            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0.0

                # 使用 tqdm 显示进度条
                for npz_file in tqdm(self.npz_files, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                    # 加载一个 npz 文件并创建数据加载器
                    dataset = ChessDataset(os.path.join(self.npz_files_dir, npz_file))
                    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                    for inputs, labels in dataloader:
                        inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        outputs = outputs.view(-1, 64)
                        loss = self.criterion(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                        total_loss += loss.item()

                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(self.npz_files):.4f}")

            print("Training complete.")
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print("Model saved.")

if __name__ == "__main__":
    # Data processing
    processor = ChessDataPreprocessor("chess_games.csv", max_steps=244)
    processor.max_moves = processor.find_optimal_max_steps()
    processor.prepare_and_save_data_in_batches()
    # train_data, train_labels = processor.prepare_and_save_data_in_batches()
    

    # # Model and training
    mcts_cnn_model = ChessModel(conv_size=32, conv_depth=3, num_classes=64)
    trainer = ChessTrainer(mcts_cnn_model, data_dir='output_mcts_cnn', epochs=20, batch_size=16, lr=0.001)
    trainer.train()
    trainer.save_model("mcts_cnn_model.pth")

    # print("模型保存完成。")

    # # Validate the model
    # validator = ChessModelValidator(trainer.model, train_data, train_labels)
    # validator.validate()
