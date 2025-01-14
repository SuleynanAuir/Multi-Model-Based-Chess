import os
import time
import chess
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from battle import (
    board_state_to_input_model1,
    board_state_to_input_model2,
    load_algorithm1_model,
    load_algorithm2_model,
)

from MCTS_module import MCTS_CNN_Model, load_model_with_missing_layers  # Ensure this module exists
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACTION_SIZE = 64 * 64  # Total possible actions in chess (from_square * 64 + to_square)

# ============================
# DQN Model Definition
# ============================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # For matching dimensions

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)
        return out

class DQN(nn.Module):
    def __init__(self, num_actions=ACTION_SIZE):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.resblock1 = ResidualBlock(64, 128, stride=1, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
        ))
        self.resblock2 = ResidualBlock(128, 128)
        self.resblock3 = ResidualBlock(128, 256, stride=1, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        ))

        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)

        out = self.flatten(out)
        out = self.fc(out)
        return out

# ============================
# Utility Functions
# ============================
from math import log, sqrt

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

class Node:
    def __init__(self, state, move=None, parent=None):
        self.move = move
        self.state = state
        self.parent = parent
        self.unexplored_moves = list(state.legal_moves)  # Unexplored legal moves
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, state, move):
        child_node = Node(state, move, self)
        self.children.append(child_node)
        self.unexplored_moves.remove(move)
        return child_node

    def UCT_select_child(self, model):
        best_value = float('-inf')
        best_child = None
        for child in self.children:
            win_rate = child.wins / (child.visits + 1e-6)  # Avoid division by zero
            exploration = sqrt(2 * log(self.visits + 1) / (child.visits + 1e-6))
            heuristic = self.heuristic(child.state, model)
            value = win_rate + exploration + heuristic

            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    @staticmethod
    def split_dims(board):
        board3d = np.zeros((12, 8, 8), dtype=np.int8)
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                row, col = divmod(square, 8)
                board3d[piece - 1, row, col] = 1
            for square in board.pieces(piece, chess.BLACK):
                row, col = divmod(square, 8)
                board3d[5 + piece, row, col] = 1
        return board3d

    def heuristic(self, state, model):
        state_tensor = torch.tensor(Node.split_dims(state), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)  # CNN model outputs move probabilities
        return torch.max(q_values).item()

    def update(self, result):
        self.visits += 1
        self.wins += result

class MCTS_CNN_Model(nn.Module):
    def __init__(self, conv_size, conv_depth, num_classes):
        super(MCTS_CNN_Model, self).__init__()
        self.cnn_model = ChessModel(conv_size=conv_size, conv_depth=conv_depth, num_classes=num_classes)
        self.rootnode = None

    def set_root_node(self, rootstate):
        self.rootnode = Node(state=rootstate)

    def forward(self, x):
        return self.cnn_model(x)

    def select_move_based_on_policy(self, policy_probs):
        legal_moves = list(self.rootnode.state.legal_moves)  # Get legal moves
        if len(policy_probs) < len(legal_moves):
            # If policy_probs has fewer elements, adjust accordingly
            legal_move_probs = [policy_probs[i] if i < len(policy_probs) else 0 for i in range(len(legal_moves))]
        else:
            legal_move_probs = [policy_probs[i] for i in range(len(legal_moves))]
        selected_index = random.choices(range(len(legal_moves)), weights=legal_move_probs, k=1)[0]  # Choose based on probabilities
        return selected_index

    def UCT_search(self, itermax, depthmax):
        rootnode = self.rootnode
        for i in range(itermax):
            node = rootnode
            state = rootnode.state.copy()

            while node.unexplored_moves == [] and node.children != []:
                node = node.UCT_select_child(self.cnn_model)
                state.push(node.move)

            if node.unexplored_moves != []:
                m = random.choice(node.unexplored_moves)
                state.push(m)
                node = node.add_child(state, m)

            while state.legal_moves != []:
                move_probs = torch.softmax(self.cnn_model(Node.split_dims(state)), dim=-1).squeeze()
                move_index = self.select_move_based_on_policy(move_probs)
                legal_moves = list(state.legal_moves)
                state.push(legal_moves[move_index])

            result = 1 if self.cnn_model(Node.split_dims(state)).item() > 0 else 0
            node.update(result)

        return self.select_move_based_on_policy(
            torch.softmax(self.cnn_model(Node.split_dims(self.rootnode.state)), dim=-1)
        )

# ============================
# Model Loading and Initialization
# ============================

def load_model_with_missing_layers(model, checkpoint):
    try:
        # Use strict=False to allow missing layers during loading
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    # Handle missing layers
    missing_layers = set(model.state_dict().keys()) - set(checkpoint['model_state_dict'].keys())
    if missing_layers:
        print("Warning: Missing layers detected. Initializing missing layers.")
        for layer in missing_layers:
            if 'conv_layers' in layer:
                print(f"Reinitializing missing layer: {layer}")
                # Reinitialize missing convolutional layers
                model.cnn_model.conv_layers.apply(init_weights)
            else:
                print(f"Layer {layer} is missing and will be initialized.")
                # Reinitialize other missing layers
                model.apply(init_weights)

def init_weights(layer):
    """Initialize weights for convolutional and linear layers."""
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def action_to_move(action):
    from_sq = action // 64
    to_sq = action % 64
    return from_sq, to_sq

def board_state_to_rl_input(board):
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

def load_dqn_model(file_path, num_actions=ACTION_SIZE):
    dqn = DQN(num_actions=num_actions).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    dqn.load_state_dict(checkpoint)
    dqn.eval()
    return dqn

def load_algorithm1_model(file_path, num_classes=64):
    model = MCTS_CNN_Model(conv_size=32, conv_depth=3, num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    load_model_with_missing_layers(model, checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model, device

def load_algorithm2_model(file_path, num_classes=80):
    class ChessCNN(torch.nn.Module):
        def __init__(self, num_classes=80):
            super(ChessCNN, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveMaxPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            return self.model(x)

    model = ChessCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# ============================
# Evaluation and Logging
# ============================

def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    white_score = 0
    black_score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_score += piece_value
            else:
                black_score += piece_value

    return white_score, black_score

def log_game_state(board, white_score, black_score, move_number, file_path="game_log3.txt"):
    with open(file_path, "a") as file:
        file.write(f"Move {move_number}:\n")
        file.write(str(board) + "\n")
        file.write(f"White score: {white_score}\n")
        file.write(f"Black score: {black_score}\n")
        file.write("-" * 4 + "\n")

def print_board_state(board, white_score, black_score, move_number):
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)
    # time.sleep(1)  # Uncomment to add delay

def game_over(board):
    return board.is_game_over()

def get_legal_moves(board):
    """
    获取所有合法的棋步，返回 chess.Move 对象列表。
    """
    return list(board.legal_moves)

# ============================
# Modified predict_best_move_model1 Function
# ============================

def predict_best_move_model1(model, board_state):
    model = model.to(device)
    legal_moves = get_legal_moves(board_state)  # 获取所有合法棋步
    if not legal_moves:
        return None

    board_input = board_state_to_input_model1(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).squeeze().cpu().numpy()

    # print(f"Model output shape: {move_probs.shape}")  # Debugging line

    if len(move_probs) != 64:
        raise ValueError(f"Expected move_probs of size 64, but got {len(move_probs)}")

    # Step 1: Select "from" square based on move_probs
    from_square_probs = move_probs  # Size 64
    from_square_probs = from_square_probs / from_square_probs.sum()  # Normalize

    # Create a list of legal "from" squares
    legal_from_squares = [move.from_square for move in legal_moves]
    # Extract probabilities for legal "from" squares
    legal_from_probs = [from_square_probs[square] for square in legal_from_squares]
    # Normalize the probabilities
    legal_from_probs = np.array(legal_from_probs)
    legal_from_probs = legal_from_probs / legal_from_probs.sum()

    # Step 2: Select "from" square based on probabilities
    selected_from_index = np.random.choice(len(legal_from_squares), p=legal_from_probs)
    selected_from_square = legal_from_squares[selected_from_index]

    # Get all legal moves from the selected "from" square
    selected_moves = [move for move in legal_moves if move.from_square == selected_from_square]
    if not selected_moves:
        return None  # No moves from the selected square

    # If multiple moves from the same "from" square, select randomly
    best_move = random.choice(selected_moves)
    return best_move

# ============================
# Other predict_best_move Functions
# ============================

def predict_best_move_model2(model, board_state):
    model = model.to(device)
    legal_moves = get_legal_moves(board_state)
    if not legal_moves:
        return None

    best_move = None
    best_move_value = -float('inf')  # Initialize to negative infinity

    for move in legal_moves:
        board_input = board_state_to_input_model2(board_state)
        board_input = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(board_input)

        move_index = legal_moves.index(move)
        move_value = prediction[0, move_index].item()

        if move_value > best_move_value:
            best_move_value = move_value
            best_move = move

    return best_move

def predict_best_move_dqn(model, board_state, epsilon=0.5):
    """
    使用 DQN 模型预测最佳棋步。
    在评估期间，将 epsilon 设置为 0 以专注于利用。
    """
    legal_moves = get_legal_moves(board_state)
    if not legal_moves:
        return None

    board_input = board_state_to_rl_input(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        q_values = model(board_input).squeeze().cpu().numpy()

    # 将合法棋步转换为动作
    legal_actions = [move.from_square * 64 + move.to_square for move in legal_moves]

    if random.random() < epsilon:
        # 探索：选择随机合法动作
        chosen_action = random.choice(legal_actions)
    else:
        # 利用：选择 Q 值最高的动作
        legal_q_values = q_values[legal_actions]
        chosen_action = legal_actions[np.argmax(legal_q_values)]

    from_sq, to_sq = action_to_move(chosen_action)
    return chess.Move(from_sq, to_sq)

# ============================
# Apply Move Function
# ============================

def apply_move(board, move):
    """
    应用棋步到棋盘。
    move 可以是字符串（UCI 格式）或 chess.Move 对象。
    """
    chess_move = chess.Move.from_uci(move) if isinstance(move, str) else move
    if chess_move in board.legal_moves:
        board.push(chess_move)

# ============================
# Game Playing Function
# ============================

def play_game(agent_white, agent_white_type, agent_black, agent_black_type, game_number, save_dir="game_scores/agents_tournament"):
    """
    玩一局游戏，两个代理之间的对弈。

    参数:
    - agent_white: 白方代理模型
    - agent_white_type: 白方代理类型（例如 "cnn_mcts", "cnn", "dqn"）
    - agent_black: 黑方代理模型
    - agent_black_type: 黑方代理类型
    - game_number: 当前游戏编号（用于日志）
    - save_dir: 保存游戏日志和图表的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    board = chess.Board()
    move_number = 1
    white_scores = []
    black_scores = []
    epsilon = 0  # 在评估期间设置 epsilon=0 以专注于利用

    while not game_over(board):
        white_score, black_score = evaluate_board(board)
        white_scores.append(white_score)
        black_scores.append(black_score)
        # print_board_state(board, white_score, black_score, move_number)  # Optional print

        # 白方行动
        if agent_white_type == "cnn_mcts":
            move = predict_best_move_model1(agent_white, board)
        elif agent_white_type == "cnn":
            move = predict_best_move_model2(agent_white, board)
        elif agent_white_type == "dqn":
            move = predict_best_move_dqn(agent_white, board, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown agent type: {agent_white_type}")

        if move is None:
            break

        apply_move(board, move)
        log_game_state(board, white_score, black_score, move_number)
        move_number += 1

        if game_over(board):
            break

        # 黑方行动
        if agent_black_type == "cnn_mcts":
            move = predict_best_move_model1(agent_black, board)
        elif agent_black_type == "cnn":
            move = predict_best_move_model2(agent_black, board)
        elif agent_black_type == "dqn":
            move = predict_best_move_dqn(agent_black, board, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown agent type: {agent_black_type}")

        if move is None:
            break

        apply_move(board, move)
        log_game_state(board, white_score, black_score, move_number)
        move_number += 1

    # 绘制分数进展
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(white_scores) + 1), white_scores, label="White (" + agent_white_type + ")", marker='o')
    plt.plot(range(1, len(black_scores) + 1), black_scores, label="Black (" + agent_black_type + ")", marker='x')
    plt.title(f"Game {game_number} Score Progression")
    plt.xlabel("Move Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # 保存图表
    plot_path = os.path.join(save_dir, f"game_{game_number}_scores.png")
    plt.savefig(plot_path)
    plt.close()

    return board.result()

# ============================
# Tournament Function
# ============================

def run_tournament(agent_white, agent_white_type, agent_black, agent_black_type, games=30, start_game_number=1):
    """
    运行一个锦标赛，其中两个指定的代理对弈多场。

    参数:
    - agent_white: 白方代理模型
    - agent_white_type: 白方代理类型
    - agent_black: 黑方代理模型
    - agent_black_type: 黑方代理类型
    - games: 要玩的游戏数量
    - start_game_number: 开始的游戏编号（用于日志）

    返回:
    - results: 包含 "1-0", "0-1", 和 "1/2-1/2" 数量的字典
    """
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    save_dir = "game_scores/agents_tournament"

    for game in range(games):
        current_game_number = start_game_number + game
        try:
            result = play_game(agent_white, agent_white_type, agent_black, agent_black_type, game_number=current_game_number, save_dir=save_dir)
            results[result] += 1
            print(f"Game {current_game_number}: Result = {result}")
        except Exception as e:
            print(f"Game {current_game_number} failed with error: {e}")

    return results

# ============================
# Plotting Function
# ============================

def plot_win_rate(results, labels, save_dir="game_scores/agents_tournament", filename="tournament_results.png"):
    """
    绘制胜率结果。

    参数:
    - results: 包含 "1-0", "0-1", 和 "1/2-1/2" 数量的字典
    - labels: [白方代理类型, 黑方代理类型]
    - save_dir: 保存图表的目录
    - filename: 图表文件名
    """
    white_wins = results["1-0"]
    black_wins = results["0-1"]
    draws = results["1/2-1/2"]

    # 根据代理类型确定 DQN 的胜负
    # 如果白方是 DQN，"1-0" 表示 DQN 赢
    # 如果黑方是 DQN，"0-1" 表示 DQN 赢
    if labels[0] == "dqn":
        dqn_wins = white_wins
        dqn_losses = black_wins
    elif labels[1] == "dqn":
        dqn_wins = black_wins
        dqn_losses = white_wins
    else:
        dqn_wins = 0
        dqn_losses = 0

    dqn_draws = draws

    labels_bar = ['DQN Wins', 'DQN Losses', 'Draws']
    win_counts = [dqn_wins, dqn_losses, dqn_draws]

    # 绘制条形图
    fig, ax = plt.subplots()
    colors = ['green', 'red', 'blue']
    ax.bar(labels_bar, win_counts, color=colors)

    # 添加标题和标签
    ax.set_title(f'Tournament Results: {labels[0]} (White) vs {labels[1]} (Black)')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Number of Games')

    # 添加胜率
    total_games = sum(win_counts)
    if total_games > 0:
        win_rates = [round((count / total_games) * 100, 2) for count in win_counts]
    else:
        win_rates = [0, 0, 0]

    for i, win_rate in enumerate(win_rates):
        ax.text(i, win_counts[i] + 0.5, f'{win_rate}%', ha='center', va='bottom')

    # 保存图表
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close(fig)

# ============================
# Main Function
# ============================

def main():
    # 加载现有模型
    print("Loading Model 1 (MCTS CNN)...")
    algorithm1, device1 = load_algorithm1_model("mcts_cnn_model.pth", num_classes=64)

    print("Loading Model 2 (Chess CNN QL)...")
    algorithm2 = load_algorithm2_model("chess_cnn_model_ql.pth", num_classes=80)

    # 加载训练好的 DQN 模型
    print("Loading DQN Model...")
    dqn = load_dqn_model("rl_dqn_chess_final.pth", num_actions=ACTION_SIZE)

    # 定义对弈对
    matchups = [
        {
            "agent_white": dqn,
            "agent_white_type": "dqn",
            "agent_black": algorithm1,
            "agent_black_type": "cnn_mcts",
            "games": 500,
            "description": "DQN (White) vs MCTS-CNN (Black)",
            "filename": "dqn_vs_mctscnn.png"
        },
        {
            "agent_white": algorithm2,
            "agent_white_type": "cnn",
            "agent_black": dqn,
            "agent_black_type": "dqn",
            "games": 500,
            "description": "Chess-CNN (White) vs DQN (Black)",
            "filename": "cnn_vs_dqn.png"
        }
    ]

    total_results = {}

    start_game_number = 1
    for matchup in matchups:
        print(f"\nStarting Tournament: {matchup['description']} for {matchup['games']} games...")
        results = run_tournament(
            agent_white=matchup["agent_white"],
            agent_white_type=matchup["agent_white_type"],
            agent_black=matchup["agent_black"],
            agent_black_type=matchup["agent_black_type"],
            games=matchup["games"],
            start_game_number=start_game_number
        )
        total_results[matchup["description"]] = results
        plot_win_rate(
            results=results,
            labels=[matchup["agent_white_type"], matchup["agent_black_type"]],
            save_dir="game_scores/agents_tournament",
            filename=matchup["filename"]
        )
        start_game_number += matchup["games"]

    # 输出最终结果
    for description, results in total_results.items():
        print(f"\nFinal Results for {description}: {results}")
    print("Tournament completed. Results have been plotted and saved.")

if __name__ == "__main__":
    main()



