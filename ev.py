import os
import sys
import chess
import matplotlib
import torch
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm  # For progress bar during training
import matplotlib.pyplot as plt  # For visualization
import pickle  # Ensure pickle is imported
import random
import math
import torch.nn as nn

from battle_dqn import board_state_to_rl_input
matplotlib.use('Agg')
# ---------------------------- Device Configuration ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------- QLearning_Algorithm Class ----------------------------

class QLearning_Algorithm:
    def __init__(self, q_table_path="qlearning_agent.pkl"):
        """
        初始化 Q-learning 代理。
        """
        self.q_table = defaultdict(partial(defaultdict, float))  # 使用 defaultdict 来存储 Q 表
        self.q_table_path = q_table_path
        self.load_q_table(self.q_table_path)

    def get_q_value(self, state, action):
        """
        获取给定状态和动作的 Q 值。
        """
        return self.q_table[state][action]

    def choose_action(self, board, epsilon=0.1):
        """
        基于 epsilon-greedy 策略选择动作。
        """
        state = board.fen()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if random.random() < epsilon:
            # 探索：选择随机合法动作
            return random.choice(legal_moves).uci()
        
        # 从合法动作中选择 Q 值最高的动作
        q_values = [(move, self.get_q_value(state, move.uci())) for move in legal_moves]
        best_move = max(q_values, key=lambda x: x[1])[0]
        return best_move.uci()

    def save_q_table(self, filename):
        """
        将 Q 表保存到文件。
        """
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename):
        """
        从文件加载 Q 表。
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(partial(defaultdict, float), data)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved Q-table found at {filename}, starting with an empty Q-table.")

# ---------------------------- AI Model Loading Functions ----------------------------

def board_state_to_input_model1(board):
    """
    将棋盘状态转换为模型1的输入格式：[batch_size, steps, channels, height, width]
    """
    board_input = np.zeros((8, 8, 12))  # 8x8棋盘，12个通道
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            channel = piece_type - 1 + (6 if piece_color == chess.BLACK else 0)  # 0-5为白棋，6-11为黑棋
            row, col = divmod(square, 8)
            board_input[row, col, channel] = 1  # 标记该位置有棋子

    # 转换为 [batch_size, steps, channels, height, width]
    board_input = np.transpose(board_input, (2, 0, 1))  # (channels, height, width)
    board_input = np.expand_dims(board_input, axis=0)  # 添加 batch_size 维度
    board_input = np.expand_dims(board_input, axis=1)  # 添加 steps 维度
    return board_input

def board_state_to_input_model2(board):
    """
    将棋盘状态转换为模型2的输入格式：[batch_size, channels, height, width]
    """
    board_input = np.zeros((8, 8, 12))  # 8x8棋盘，12个通道
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            channel = piece_type - 1 + (6 if piece_color == chess.BLACK else 0)  # 0-5为白棋，6-11为黑棋
            row, col = divmod(square, 8)
            board_input[row, col, channel] = 1  # 标记该位置有棋子
    return board_input

def load_algorithm1_model(file_path, num_classes=64):
    """
    加载模型1（MCTS-CNN）。
    """
    from MCTS_module import MCTS_CNN_Model, load_model_with_missing_layers
    model = MCTS_CNN_Model(conv_size=32, conv_depth=3, num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    load_model_with_missing_layers(model, checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def load_algorithm2_model(file_path, num_classes=80):
    """
    加载模型2（Chess-CNN）。
    """
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
            x = x.permute(0, 3, 1, 2)  # 从 [batch, height, width, channels] 转换为 [batch, channels, height, width]
            return self.model(x)

    model = ChessCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def load_dqn_model(file_path, num_actions=64*64):
    """
    加载 DQN 模型。
    """
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
            self.downsample = downsample  # 用于匹配维度

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            # 如果需要下采样，则调整identity的维度
            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity  # 残差连接
            out = self.relu(out)
            return out

    class DQN(nn.Module):
        def __init__(self, num_actions=64*64):
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            # 添加残差块
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

            # 全连接层
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

    dqn = DQN(num_actions=num_actions).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    dqn.load_state_dict(checkpoint)
    dqn.eval()
    return dqn

# ---------------------------- Move Prediction Functions ----------------------------

def get_legal_moves(board):
    """
    获取当前棋盘状态的所有合法走法，使用 UCI 格式。
    """
    return [move.uci() for move in board.legal_moves]

def predict_best_move_model1(model, board_state):
    """
    使用模型1（MCTS-CNN）预测最佳走法。
    """
    model = model.to(device)
    legal_moves = get_legal_moves(board_state)  # 获取所有合法走法
    if not legal_moves:
        return None

    # 获取模型输入
    board_input = board_state_to_input_model1(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 模型输出 logits
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).squeeze().cpu().numpy()

    # 将合法走法映射到它们的概率
    legal_move_probs = {}
    for i, move in enumerate(legal_moves):
        if i < len(move_probs):
            legal_move_probs[move] = move_probs[i]
        else:
            legal_move_probs[move] = 0  # 如果超出范围，赋值为0概率

    # 选择概率最高的走法
    if legal_move_probs:
        best_move = max(legal_move_probs, key=legal_move_probs.get)
        return best_move
    return None

def predict_best_move_model2(model, board_state):
    """
    使用模型2（Chess-CNN）预测最佳走法。
    """
    model = model.to(device)
    legal_moves = get_legal_moves(board_state)

    if not legal_moves:
        return None  # 如果没有合法走法，返回 None

    best_move = None
    best_move_value = -float('inf')  # 存储最佳走法的值

    # 将棋盘状态转换为模型输入
    board_input = board_state_to_input_model2(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为 PyTorch 张量

    with torch.no_grad():
        prediction = model(board_input)  # 获取预测结果

    move_values = prediction[0].cpu().numpy()

    # 根据模型输出的值为每个合法走法赋值
    for idx, move in enumerate(legal_moves):
        if idx < len(move_values):
            move_value = move_values[idx]
            if move_value > best_move_value:
                best_move_value = move_value
                best_move = move

    return best_move

def predict_best_move_dqn(model, board_state, epsilon=0.0):
    """
    使用 DQN 模型预测最佳走法。
    在评估期间，将 epsilon 设置为 0 以专注于利用。
    """
    legal_moves = get_legal_moves(board_state)
    if not legal_moves:
        return None

    board_input = board_state_to_rl_input(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        q_values = model(board_input).squeeze().cpu().numpy()

    # 将合法棋步转换为动作索引
    legal_actions = []
    for move_str in legal_moves:
        try:
            move_obj = chess.Move.from_uci(move_str)
            action = move_obj.from_square * 64 + move_obj.to_square
            legal_actions.append(action)
        except ValueError:
            print(f"遇到无效的走法字符串: {move_str}")
            continue  # 跳过无效的走法字符串

    if not legal_actions:
        return None  # 转换后没有有效的动作

    legal_actions = np.array(legal_actions)

    if random.random() < epsilon:
        # 探索：选择随机合法动作
        chosen_action = random.choice(legal_actions)
    else:
        # 利用：选择 Q 值最高的动作
        q_values_legal = q_values[legal_actions]
        chosen_action = legal_actions[np.argmax(q_values_legal)]

    from_sq, to_sq = action_to_move(chosen_action)
    return chess.Move(from_sq, to_sq).uci()

def action_to_move(action):
    from_sq = action // 64
    to_sq = action % 64
    return from_sq, to_sq

def apply_move(board, move):
    """
    应用一个走法到棋盘上。
    """
    if isinstance(move, str):
        try:
            chess_move = chess.Move.from_uci(move)
        except ValueError:
            print(f"无效的走法字符串: {move}")
            return
    elif isinstance(move, chess.Move):
        chess_move = move
    else:
        raise ValueError("Invalid move type. Move must be a string or chess.Move object.")
    
    if chess_move in board.legal_moves:
        board.push(chess_move)
    else:
        print(f"Attempted invalid move: {chess_move}")

# ---------------------------- Game Play Functions ----------------------------

def evaluate_board(board):
    """
    评估棋盘并返回白方和黑方的分数。
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    white_score = 0
    black_score = 0

    for piece_type in piece_values:
        white_score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        black_score += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    return white_score, black_score

def print_board_state(board, white_score, black_score, move_number):
    """
    打印当前棋盘状态以及分数和走法编号。
    """
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)

def game_over(board):
    """
    检查游戏是否结束。
    """
    return board.is_game_over()

def play_game(player1, player2, game_number, player1_name, player2_name, save_dir):
    """
    进行一局游戏。
    
    参数:
    - player1: 白方玩家的走法选择函数。
    - player2: 黑方玩家的走法选择函数。
    - game_number: 游戏编号，用于图表命名。
    - player1_name: 白方使用的算法名称。
    - player2_name: 黑方使用的算法名称。
    - save_dir: 保存游戏图片的目录。
    
    返回:
    - result: 游戏结果 ("1-0", "0-1", "1/2-1/2")。
    - final_white_score: 白方的最终分数。
    - final_black_score: 黑方的最终分数。
    """
    board = chess.Board()
    move_number = 1
    white_scores = []
    black_scores = []
    
    while not game_over(board):
        white_score, black_score = evaluate_board(board)
        white_scores.append(white_score)
        black_scores.append(black_score)
        print_board_state(board, white_score, black_score, move_number)
        
        if board.turn == chess.WHITE:
            move = player1(board)
        else:
            move = player2(board)
        
        if move is None:
            break  # 如果没有合法走法，直接结束对局
        apply_move(board, move)
        
        move_number += 1

    # 确定游戏结果
    result = board.result()
    final_white_score, final_black_score = evaluate_board(board)

    # 绘制分数进展图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(white_scores) + 1), white_scores, label=f"White ({player1_name})", marker='o')
    plt.plot(range(1, len(black_scores) + 1), black_scores, label=f"Black ({player2_name})", marker='x')
    plt.title(f"Game {game_number} Score Progression")
    plt.xlabel("Move Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图表
    plot_path = os.path.join(save_dir, f"game_{game_number}_scores_{player1_name}_vs_{player2_name}.png")
    plt.savefig(plot_path)
    plt.close()
    
    return result, final_white_score, final_black_score  # 返回游戏结果及最终分数

# ---------------------------- Tournament Functions ----------------------------

def run_tournament(q_agent, algorithm1, algorithm2, dqn_agent, games_per_matchup=50):
    """
    运行一个锦标赛，其中 Minimax 代理与其他模型对弈。
    
    参数:
    - q_agent: Q-learning 代理实例。
    - algorithm1: MCTS-CNN 模型。
    - algorithm2: Chess-CNN 模型。
    - dqn_agent: DQN 模型。
    - games_per_matchup: 每种对局方式的游戏数量。
    
    返回:
    - overall_results: 包含各对局方式的胜负平结果的字典。
    - overall_scores: 包含各对局方式的总分数差的字典（后续计算平均分数差）。
    """
    overall_results = {
        "Minimax_vs_CNN": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "Minimax_vs_CNN-MCTS": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "Minimax_vs_Q-learning": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "Minimax_vs_DQN": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
    }
    
    overall_scores = {
        "Minimax_vs_CNN": {"score_diff_sum": 0},
        "Minimax_vs_CNN-MCTS": {"score_diff_sum": 0},
        "Minimax_vs_Q-learning": {"score_diff_sum": 0},
        "Minimax_vs_DQN": {"score_diff_sum": 0},
    }
    
    # 定义玩家函数
    def player_minimax(board):
        minimax_ai = MinimaxAI(depth=3)  # 定义 Minimax AI 的深度
        return minimax_ai.get_best_move(board)
    
    def player_cnn(board):
        return predict_best_move_model1(algorithm1, board)
    
    def player_cnn_mcts(board):
        return predict_best_move_model1(algorithm1, board)  # 假设 MCTS-CNN 使用相同的预测函数
    
    def player_q_learning(board):
        return q_agent.choose_action(board, epsilon=0.15)  # 评估期间设置 epsilon=0
    
    def player_dqn(board):
        return predict_best_move_dqn(dqn_agent, board, epsilon=0.05)
    
    # 对局安排
    matchups = [
        ("Minimax_vs_CNN", player_minimax, player_cnn, "Minimax", "CNN"),
        ("Minimax_vs_CNN-MCTS", player_minimax, player_cnn_mcts, "Minimax", "CNN-MCTS"),
        ("Minimax_vs_Q-learning", player_minimax, player_q_learning, "Minimax", "Q-learning"),
        ("Minimax_vs_DQN", player_minimax, player_dqn, "Minimax", "DQN"),
    ]
    
    for matchup_name, player1, player2, player1_algo, player2_algo in matchups:
        print(f"\n=== Starting Matchup: {matchup_name} ===")
        matchup_dir = os.path.join("game_scores", "tournament", matchup_name)
        os.makedirs(matchup_dir, exist_ok=True)
        for game in tqdm(range(1, games_per_matchup + 1), desc=matchup_name):
            game_number = f"{matchup_name}_Game_{game}"
            result, final_white_score, final_black_score = play_game(
                player1, player2, game_number=game_number, 
                player1_name=player1_algo, player2_name=player2_algo, 
                save_dir=matchup_dir
            )
            overall_results[matchup_name][result] += 1
            # 计算分数差异，白方为 Minimax
            overall_scores[matchup_name]["score_diff_sum"] += (final_white_score - final_black_score)
        print(f"=== Completed Matchup: {matchup_name} ===")
    
    # 计算每种对局方式的平均分数差
    overall_scores_avg = {}
    for matchup in overall_scores:
        avg_diff = overall_scores[matchup]["score_diff_sum"] / games_per_matchup
        overall_scores_avg[matchup] = {"average_score_diff": avg_diff}
    
    return overall_results, overall_scores_avg

# ---------------------------- Plotting Function ----------------------------

def plot_win_rate(results, overall_scores_avg):
    """
    绘制锦标赛的胜率统计图和平均分数差异图。
    """
    matchup_names = list(results.keys())
    white_wins = [results[matchup]["1-0"] for matchup in matchup_names]
    black_wins = [results[matchup]["0-1"] for matchup in matchup_names]
    draws = [results[matchup]["1/2-1/2"] for matchup in matchup_names]
    average_score_diffs = [overall_scores_avg[matchup]["average_score_diff"] for matchup in matchup_names]

    # 绘制胜率计数
    x = np.arange(len(matchup_names))
    width = 0.2

    plt.figure(figsize=(14, 8))
    
    plt.bar(x - width, white_wins, width, label='White Wins', color='green')
    plt.bar(x, black_wins, width, label='Black Wins', color='red')
    plt.bar(x + width, draws, width, label='Draws', color='blue')

    plt.xlabel('Matchups')
    plt.ylabel('Number of Games')
    plt.title('Tournament Results: Win/Loss/Draw Counts')
    plt.xticks(x, matchup_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')

    # 保存胜率计数图
    plt.savefig(os.path.join("game_scores", "tournament", "tournament_win_loss_draw_counts.png"))
    plt.close()

    # 绘制平均分数差异
    plt.figure(figsize=(14, 8))
    plt.bar(matchup_names, average_score_diffs, color='purple')
    plt.xlabel('Matchups')
    plt.ylabel('Average Score Difference (White - Black)')
    plt.title('Tournament Results: Average Score Difference per Game')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')

    # 保存平均分数差异图
    plt.savefig(os.path.join("game_scores", "tournament", "tournament_average_score_diff.png"))
    plt.close()

    # 绘制胜率百分比
    plt.figure(figsize=(14, 8))
    white_percent = [ (w / (w + b + d)) * 100 if (w + b + d) >0 else 0 for w, b, d in zip(white_wins, black_wins, draws)]
    black_percent = [ (b / (w + b + d)) * 100 if (w + b + d) >0 else 0 for w, b, d in zip(white_wins, black_wins, draws)]
    draw_percent = [ (d / (w + b + d)) * 100 if (w + b + d) >0 else 0 for w, b, d in zip(white_wins, black_wins, draws)]

    plt.bar(x - width, white_percent, width, label='White Wins (%)', color='green')
    plt.bar(x, black_percent, width, label='Black Wins (%)', color='red')
    plt.bar(x + width, draw_percent, width, label='Draws (%)', color='blue')

    plt.xlabel('Matchups')
    plt.ylabel('Percentage (%)')
    plt.title('Tournament Results: Win/Loss/Draw Percentages')
    plt.xticks(x, matchup_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')

    # 保存胜率百分比图
    plt.savefig(os.path.join("game_scores", "tournament", "tournament_win_loss_draw_percentages.png"))
    plt.close()

    # 绘制所有对局方式的平均分数差异对比图
    plt.figure(figsize=(14, 8))
    plt.bar(matchup_names, average_score_diffs, color='orange')
    plt.xlabel('Matchups')
    plt.ylabel('Average Score Difference (White - Black)')
    plt.title('Tournament Results: Average Score Difference per Game')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')

    # 在同一图中标注平均分数差异
    for i, v in enumerate(average_score_diffs):
        plt.text(i, v + (0.05 * max(average_score_diffs)), f"{v:.2f}", ha='center', va='bottom')

    # 保存最终的平均分数差异对比图
    plt.savefig(os.path.join("game_scores", "tournament", "tournament_average_score_diff_comparison.png"))
    plt.close()

    print("\nFinal Aggregated Tournament Results Plotted and Saved.")

# ---------------------------- Minimax AI Definition ----------------------------

class MinimaxAI:
    def __init__(self, depth=3):
        self.depth = depth

    def evaluate_board(self, board):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        value = 0
        for piece_type in piece_values:
            value += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            value -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        return value

    def minimax(self, board, depth, is_max):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = list(board.legal_moves)
        if is_max:
            max_eval = -math.inf
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval)
            return min_eval

    def get_best_move(self, board):
        best_move = None
        best_value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            move_value = self.minimax(board, self.depth - 1, False)
            board.pop()
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move.uci() if best_move else None

# ---------------------------- Main Execution Block ----------------------------

def main():
    # 加载模型
    try:
        print("Loading Model 1 (MCTS-CNN)...")
        algorithm1 = load_algorithm1_model("mcts_cnn_model.pth", num_classes=64)
        print("Model 1 (MCTS-CNN) loaded successfully.")
    except Exception as e:
        print(f"Error loading Model 1: {e}")
        sys.exit(1)
    
    try:
        print("Loading Model 2 (Chess-CNN)...")
        algorithm2 = load_algorithm2_model("chess_cnn_model_ql.pth", num_classes=80)
        print("Model 2 (Chess-CNN) loaded successfully.")
    except Exception as e:
        print(f"Error loading Model 2: {e}")
        sys.exit(1)
    
    try:
        print("Loading DQN Model...")
        dqn_agent = load_dqn_model("rl_dqn_chess_final.pth", num_actions=64*64)
        print("DQN Model loaded successfully.")
    except Exception as e:
        print(f"Error loading DQN Model: {e}")
        sys.exit(1)
    
    # 实例化 Q-learning 代理
    q_learning_agent = QLearning_Algorithm(q_table_path="qlearning_agent.pkl")
    
    # 实例化 Minimax AI
    minimax_ai = MinimaxAI(depth=3)
    
    # 运行锦标赛
    games_per_matchup = 15  # 每种对局方式的游戏数量，您可以根据需要调整
    print(f"\nStarting Tournament: Total {games_per_matchup * 4} Games")
    results, overall_scores_avg = run_tournament(q_learning_agent, algorithm1, algorithm2, dqn_agent, games_per_matchup=games_per_matchup)
    
    # 输出最终聚合结果
    print("\nFinal Aggregated Tournament Results:")
    for matchup, outcome in results.items():
        print(f"\nMatchup: {matchup}")
        for result, count in outcome.items():
            print(f"  {result}: {count}")
    
    # 保存 Q-learning 代理的 Q 表
    q_learning_agent.save_q_table("game_scores/tournament/qlearning_agent_after_tournament.pkl")
    
    # 绘制胜率统计图和分数差异图
    plot_win_rate(results, overall_scores_avg)
    
    print("\nTournament completed. Results and plots have been saved.")

if __name__ == "__main__":
    main()
