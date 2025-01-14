import os
import chess
import torch
import numpy as np
import random
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_action_mapping(num_classes=64):
    action_to_index = {}
    index_to_action = {}
    count = 0
    for move in chess.SQUARES:
        for target in chess.SQUARES:
            if count >= num_classes:
                break
            move_obj = chess.Move(move, target)
            action_to_index[move_obj.uci()] = count
            index_to_action[count] = move_obj.uci()
            count += 1
    return action_to_index, index_to_action

ACTION_TO_INDEX, INDEX_TO_ACTION = create_action_mapping(num_classes=64)

def get_legal_moves(board):
    return [move.uci() for move in board.legal_moves]

# 模型 1 的输入格式
def board_state_to_input_model1(board):
    board_input = np.zeros((8, 8, 12))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            channel = piece_type - 1 + (6 if piece_color == chess.BLACK else 0)
            row, col = divmod(square, 8)
            board_input[row, col, channel] = 1
    board_input = np.transpose(board_input, (2, 0, 1))
    board_input = np.expand_dims(board_input, axis=0)
    return board_input

# 模型 2 的输入格式
def board_state_to_input_model2(board):
    board_input = np.zeros((8, 8, 12))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            channel = piece_type - 1 + (6 if piece_color == chess.BLACK else 0)
            row, col = divmod(square, 8)
            board_input[row, col, channel] = 1
    board_input = np.transpose(board_input, (2, 0, 1))
    board_input = np.expand_dims(board_input, axis=0)
    return board_input

# 加载模型 1
def load_algorithm1_model(file_path, num_classes=64):
    class MCTS_CNN_Model(nn.Module):
        def __init__(self, num_classes=64):
            super(MCTS_CNN_Model, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(12, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)
    
    model = MCTS_CNN_Model(num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# 加载模型 2
def load_algorithm2_model(file_path, num_classes=80):
    class ChessCNN(nn.Module):
        def __init__(self, num_classes=80):
            super(ChessCNN, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(12, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.model(x)
    
    model = ChessCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# 模型 1 的预测
def predict_best_move_model1(model, board):
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        return None

    board_input = board_state_to_input_model1(board)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).cpu().numpy()

    # 获取合法动作对应的索引，并过滤超出模型范围的索引
    legal_move_indices = [
        ACTION_TO_INDEX[move] for move in legal_moves
        if move in ACTION_TO_INDEX and ACTION_TO_INDEX[move] < move_probs.shape[1]
    ]
    if not legal_move_indices:
        return None

    # 选择概率最高的合法动作
    best_index = legal_move_indices[np.argmax([move_probs[0][i] for i in legal_move_indices])]
    return INDEX_TO_ACTION[best_index]

# 模型 2 的预测
def predict_best_move_model2(model, board):
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        return None
    best_move = None
    best_value = -float('inf')
    for move in legal_moves:
        board_input = board_state_to_input_model2(board)
        board_input = torch.tensor(board_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = model(board_input)

        move_index = ACTION_TO_INDEX[move.uci()]
        move_value = prediction[0, move_index].item()

        if move_value > best_value:
            best_value = move_value
            best_move = move.uci()

    return best_move


# 保存模型：支持完整键值对和仅模型权重的保存
def save_model(model, optimizer, folder, filename, save_optimizer=False):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    if save_optimizer:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, file_path)
        print(f"Model and optimizer saved to {file_path}")
    else:
        torch.save(model.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")


# 加载模型
# 加载模型：支持完整键值对和仅模型权重的加载
def load_model(model_class, optimizer, file_path, num_classes, load_optimizer=False):
    model = model_class(num_classes=num_classes).to(device)
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if load_optimizer else checkpoint)
        if load_optimizer and optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model from {file_path}")
    else:
        print(f"File {file_path} does not exist. Starting with a new model.")
    model.eval()
    return model


# 模型定义
class MCTS_CNN_Model(nn.Module):
    def __init__(self, num_classes=64):
        super(MCTS_CNN_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ChessCNN(nn.Module):
    def __init__(self, num_classes=80):
        super(ChessCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
def predict_best_move(model, board,board_state_to_input_func):
    # 获取所有合法棋步
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        return None

    # 将棋盘状态转换为模型输入格式
    board_input = board_state_to_input_func(board)  # 使用特定算法的棋盘状态转换函数
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    # 模型预测 Q值
    with torch.no_grad():
        q_values = model(board_input)  # 输出所有动作的 Q值
        q_values = q_values.squeeze(0).cpu().numpy()  # 转为 numpy 数组

    # 找出合法动作中 Q值最大的动作
    legal_move_indices = [ACTION_TO_INDEX[move] for move in legal_moves if move in ACTION_TO_INDEX]
    if not legal_move_indices:
        return None

    best_index = max(legal_move_indices, key=lambda idx: q_values[idx])  # 比较合法动作的 Q值
    best_move = INDEX_TO_ACTION[best_index]
    return best_move


# 应用棋步
def apply_move(board, move):
    board.push(chess.Move.from_uci(move))

def evaluate_board_action(board, move, player_color):
    """
    根据棋步的进攻性或保守性，计算奖励值。
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    reward = 0
    move_obj = chess.Move.from_uci(move)
    if move_obj in board.legal_moves:
        # 模拟走棋
        board.push(move_obj)

        # 检查是否吃子
        captured_piece = board.piece_at(move_obj.to_square)
        if captured_piece:
            reward += piece_values.get(captured_piece.piece_type, 0)  # 按棋子价值给予奖励

        # 检查是否将军
        if board.is_check():
            reward += 7  # 将军奖励

        # 检查是否威胁到重要棋子
        attacked_squares = board.attacks(move_obj.to_square)
        for square in attacked_squares:
            target_piece = board.piece_at(square)
            if target_piece and target_piece.color != player_color:
                if target_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                    reward += 5  # 威胁重要棋子的额外奖励

        # 检查保守性：如果既没有吃子、将军，也没有威胁，则给予惩罚
        if not captured_piece and not board.is_check() and reward == 0:
            reward -= 5  # 保守棋步的惩罚

        # 回滚棋步
        board.pop()

    return reward

# 自对弈
def self_play(algorithm1, algorithm2, num_games=100):
    memory = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    total_rewards1 = []
    total_rewards2 = []

    for _ in range(num_games):
        board = chess.Board()
        current_player = 1
        game_states = []
        rewards1 = []
        rewards2 = []

        while not board.is_game_over(claim_draw=True):
            if current_player == 1:
                move = predict_best_move(algorithm2, board, board_state_to_input_model2)
                player_color = chess.WHITE
            else:
                move = predict_best_move(algorithm2, board, board_state_to_input_model2)
                player_color = chess.BLACK

            if move is None:
                break

            action_index = ACTION_TO_INDEX.get(move)
            if action_index is not None:
                state = board_state_to_input_model2(board) if current_player == 1 else board_state_to_input_model2(board)
                action_reward = evaluate_board_action(board, move, player_color)
                if current_player == 1:
                    rewards1.append(action_reward)
                else:
                    rewards2.append(action_reward)
                game_states.append((state, action_index, action_reward))

            board.push(chess.Move.from_uci(move))
            current_player = 3 - current_player

        result = board.result(claim_draw=True)
        reward_mapping = {"1-0": 100, "0-1": -100, "1/2-1/2": -50}
        final_reward = reward_mapping.get(result, 0)

        if result in results:
            results[result] += 1

        for state, action, action_reward in game_states:
            memory.append((state, action, action_reward + final_reward))

        total_rewards1.append(sum(rewards1))
        total_rewards2.append(sum(rewards2))
    return memory, results, total_rewards1, total_rewards2

# 训练模型
def train_model(model, optimizer, memory, batch_size=64):
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # 获取逐样本损失
    random.shuffle(memory)

    for i in range(0, len(memory), batch_size):
        batch = memory[i:i + batch_size]
        states, actions, rewards = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # 奖励作为张量

        predictions = model(states)
        predictions = predictions.view(-1, predictions.size(-1))  # 形状调整
        actions = actions.view(-1)
        rewards = rewards.view(-1)

        losses = loss_fn(predictions, actions)  # 逐样本损失
        adjusted_losses = losses *rewards.abs()  # 奖励调整损失
        loss = adjusted_losses.mean()  # 平均化损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 强化学习循环：不同模型保存逻辑
def reinforcement_learning_loop(algorithm1, algorithm2, optimizer1, optimizer2, num_iterations=100, save_interval=10):
    
    win_rates1 = []
    win_rates2 = []
    reward_history1 = []
    reward_history2 = []

    progress = tqdm(range(num_iterations), desc="Reinforcement Learning Iterations")
    for iteration in progress:
        memory, results, total_rewards1, total_rewards2 = self_play(algorithm1, algorithm2, num_games=10)

        total_games = sum(results.values())
        win_rate1 = results["1-0"] / total_games if total_games > 0 else 0
        win_rate2 = results["0-1"] / total_games if total_games > 0 else 0
        win_rates1.append(win_rate1)
        win_rates2.append(win_rate2)
        reward_history1.extend(total_rewards1)
        reward_history2.extend(total_rewards2)

        # train_model(algorithm1, optimizer1, memory, batch_size=256)
        train_model(algorithm2, optimizer2, memory, batch_size=256)

        progress.set_postfix({"Win Rate 1": f"{win_rate1:.2f}", "Win Rate 2": f"{win_rate2:.2f}"})

        if (iteration + 1) % save_interval == 0:
            # save_model(algorithm1, optimizer1, "models/algorithm1", f"iteration_{iteration + 1}.pth", save_optimizer=True)
            save_model(algorithm2, optimizer2, "models/algorithm2", f"iteration_{iteration + 1}.pth", save_optimizer=False)

    # save_model(algorithm1, optimizer1, "models/algorithm1", "latest.pth", save_optimizer=True)
    save_model(algorithm2, optimizer2, "models/algorithm2", "latest.pth", save_optimizer=False)
    return win_rates1, win_rates2, reward_history1, reward_history2


# 绘制胜率和奖励图
def plot_results(win_rates1, win_rates2, reward_history1, reward_history2):
    plt.figure(figsize=(12, 10))

    # 胜率图
    plt.subplot(2, 1, 1)
    plt.plot(win_rates1, label="Win Rate (Algorithm 1)", color="blue")
    plt.plot(win_rates2, label="Win Rate (Algorithm 2)", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Win Rate")
    plt.title("Win Rates Over Iterations")
    plt.legend()

    # 奖励变化图
    plt.subplot(2, 1, 2)
    plt.plot(range(len(reward_history1)), reward_history1, label="Rewards (Algorithm 1)", color="green")
    plt.plot(range(len(reward_history2)), reward_history2, label="Rewards (Algorithm 2)", color="orange")
    plt.xlabel("Game")
    plt.ylabel("Reward")
    plt.title("Reward Changes Over Games")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主程序
optimizer1 = optim.Adam(MCTS_CNN_Model(num_classes=64).parameters(), lr=0.001)
optimizer2 = optim.Adam(ChessCNN(num_classes=80).parameters(), lr=0.001)

# 第一个模型加载完整键值对，第二个仅加载模型权重
# algorithm1 = load_model(MCTS_CNN_Model, optimizer1, "models/algorithm1/latest.pth", num_classes=64, load_optimizer=True)
algorithm1 = 0
algorithm2 = load_model(ChessCNN, None, "models/algorithm2/latest.pth", num_classes=80, load_optimizer=False)

win_rates1, win_rates2, reward_history1, reward_history2 = reinforcement_learning_loop(
    algorithm1, algorithm2, optimizer1, optimizer2, num_iterations=100000, save_interval=50000
)

# 绘制结果图
plot_results(win_rates1, win_rates2, reward_history1, reward_history2)
