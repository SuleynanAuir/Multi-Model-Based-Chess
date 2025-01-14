import os
import sys
import chess
import torch
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm  # For progress bar during training
import matplotlib.pyplot as plt  # For visualization
import pickle  # 确保 pickle 被导入

# Device configuration for PyTorch
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

    def choose_action(self, board):
        """
        基于 Q 表选择动作（贪婪策略）。
        """
        state = board.fen()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # 从合法动作中选择 Q 值最高的动作
        q_values = [(move, self.get_q_value(state, move)) for move in legal_moves]
        best_move = max(q_values, key=lambda x: x[1])[0]
        return best_move

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

# ---------------------------- Model Loading Functions ----------------------------

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
    加载模型1。
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
    加载模型2。
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

# ---------------------------- Move Prediction Functions ----------------------------

def get_legal_moves(board):
    """
    获取当前棋盘状态的所有合法走法，使用 UCI 格式。
    """
    return [move.uci() for move in board.legal_moves]

def predict_best_move_model1(model, board_state):
    """
    使用模型1预测最佳走法。
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
    使用模型2预测最佳走法。
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

def apply_move(board, move):
    """
    应用一个走法到棋盘上。
    """
    if isinstance(move, str):
        chess_move = chess.Move.from_uci(move)
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
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
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

def print_board_state(board, white_score, black_score, move_number):
    """
    打印当前棋盘状态以及分数和走法编号。
    """
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)

def log_game_state(board, white_score, black_score, move_number, file_path="game_scores/cnn_cnnmcts/game_log.txt"):
    """
    将游戏状态记录到文件中。
    """
    with open(file_path, "a") as file:
        file.write(f"Move {move_number}:\n")
        file.write(str(board) + "\n")
        file.write(f"White score: {white_score}\n")
        file.write(f"Black score: {black_score}\n")
        file.write("-" * 4 + "\n")

def game_over(board):
    """
    检查游戏是否结束。
    """
    return board.is_game_over()

def play_game(player1, player2, q_agent=None, game_number=1, save_dir="game_scores/q_learning"):
    """
    进行一局游戏。
    
    参数:
    - player1: 白方玩家的走法选择函数。
    - player2: 黑方玩家的走法选择函数。
    - q_agent: 如果其中一方是 Q-learning 代理，则传入代理实例。
    - game_number: 游戏编号，用于日志和图表命名。
    - save_dir: 保存游戏日志和图表的目录。
    
    返回:
    - result: 游戏结果 ("1-0", "0-1", "1/2-1/2")。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    board = chess.Board()
    current_player = 1  # 1为白方，2为黑方
    move_number = 1
    white_scores = []
    black_scores = []
    
    while not game_over(board):
        white_score, black_score = evaluate_board(board)
        # print_board_state(board, white_score, black_score, move_number)
        white_scores.append(white_score)
        black_scores.append(black_score)
        
        if current_player == 1:
            move = player1(board)
        else:
            move = player2(board)
        
        # print(f"Legal Moves: {get_legal_moves(board)}")
        # print(f"Predicted Move: {move}")
        if move is None:
            break
        apply_move(board, move)
        
        # 如果 Q-learning 代理参与对弈且是当前玩家，记录状态和动作（用于后续分析或其他用途）
        # 由于不进行训练，省略 Q-table 更新
        
        current_player = 3 - current_player  # 切换玩家
        log_game_state(board, white_score, black_score, move_number, file_path=os.path.join(save_dir, "game_log.txt"))
        move_number += 1

    # 确定游戏结果
    result = board.result()
    
    # 绘制分数进展图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(white_scores) + 1), white_scores, label="White", marker='o')
    plt.plot(range(1, len(black_scores) + 1), black_scores, label="Black", marker='x')
    plt.title(f"Game {game_number} Score Progression")
    plt.xlabel("Move Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图表
    plot_path = os.path.join(save_dir, f"game_{game_number}_scores.png")
    plt.savefig(plot_path)
    plt.close()
    
    return result  # 返回游戏结果

# ---------------------------- Tournament Functions ----------------------------

def run_tournament(q_agent, algorithm1, algorithm2, games_per_side=10):
    
    overall_results = {
        "Q-White vs CNN-MCTS-Black": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "CNN-MCTS-White vs Q-Black": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "Q-White vs CNN-Black": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "CNN-White vs Q-Black": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
    }
    
    # 定义玩家函数
    def player_cnn_mcts(board):
        return predict_best_move_model1(algorithm1, board)
    
    def player_cnn(board):
        return predict_best_move_model2(algorithm2, board)
    
    def player_q_learning_white(board):
        return q_agent.choose_action(board)
    
    def player_q_learning_black(board):
        return q_agent.choose_action(board)
    
    # Q-learning 代理作为白方，对抗 CNN-MCTS 作为黑方
    for game in range(games_per_side):
        print(f"\nStarting Game {game + 1}: Q-White vs CNN-MCTS-Black")
        result = play_game(player_q_learning_white, player_cnn_mcts, game_number=f"QW_CNNMCTS_{game+1}")
        overall_results["Q-White vs CNN-MCTS-Black"][result] += 1
    
    # CNN-MCTS 作为白方，对抗 Q-learning 代理作为黑方
    for game in range(games_per_side):
        print(f"\nStarting Game {game + 1}: CNN-MCTS-White vs Q-Black")
        result = play_game(player_cnn_mcts, player_q_learning_black, game_number=f"CNNMCTSW_QB_{game+1}")
        overall_results["CNN-MCTS-White vs Q-Black"][result] += 1
    
    # Q-learning 代理作为白方，对抗 CNN 作为黑方
    for game in range(games_per_side):
        print(f"\nStarting Game {game + 1}: Q-White vs CNN-Black")
        result = play_game(player_q_learning_white, player_cnn, game_number=f"QW_CNN_{game+1}")
        overall_results["Q-White vs CNN-Black"][result] += 1
    
    # CNN 作为白方，对抗 Q-learning 代理作为黑方
    for game in range(games_per_side):
        print(f"\nStarting Game {game + 1}: CNN-White vs Q-Black")
        result = play_game(player_cnn, player_q_learning_black, game_number=f"CNNW_QB_{game+1}")
        overall_results["CNN-White vs Q-Black"][result] += 1
    
    return overall_results

def plot_win_rate(results):
    """
    绘制锦标赛的胜率统计图。
    """
    for matchup, outcome in results.items():
        # 提取胜、负、平局的数量
        white_wins = outcome["1-0"]
        black_wins = outcome["0-1"]
        draws = outcome["1/2-1/2"]
        
        labels = ['White Wins', 'Black Wins', 'Draws']
        win_counts = [white_wins, black_wins, draws]
        
        fig, ax = plt.subplots()
        ax.bar(labels, win_counts, color=['green', 'red', 'blue'])
        
        ax.set_title(f'Tournament Results: {matchup}')
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Number of Games')
        
        total_games = sum(win_counts)
        win_rates = [round((count / total_games) * 100, 2) for count in win_counts]
        
        for i, win_rate in enumerate(win_rates):
            ax.text(i, win_counts[i] + 0.1, f'{win_rate}%', ha='center', va='bottom')
        
        # 保存图表
        plot_path = os.path.join("game_scores/cnn_cnnmcts", f"tournament_{matchup.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close(fig)

# ---------------------------- Main Execution Block ----------------------------

if __name__ == "__main__":
    # 加载模型
    try:
        algorithm1 = load_algorithm1_model("mcts_cnn_model.pth", num_classes=64)
        print("Model 1 (MCTS-CNN) loaded successfully.")
    except Exception as e:
        print(f"Error loading Model 1: {e}")
        sys.exit(1)
    
    try:
        algorithm2 = load_algorithm2_model("chess_cnn_model.pth", num_classes=80)
        print("Model 2 (Chess-CNN) loaded successfully.")
    except Exception as e:
        print(f"Error loading Model 2: {e}")
        sys.exit(1)

    # 实例化 Q-learning 代理
    q_learning_agent = QLearning_Algorithm(q_table_path="qlearning_agent.pkl")
    
    # 运行锦标赛
    total_games = 50  # 每种对局方式的游戏数量
    print(f"\nStarting Tournament: {total_games * 4} Games in Total")
    results = run_tournament(q_learning_agent, algorithm1, algorithm2, games_per_side=total_games)

    # 输出最终聚合结果
    print("\nFinal Aggregated Tournament Results:")
    for matchup, outcome in results.items():
        print(f"\nMatchup: {matchup}")
        for result, count in outcome.items():
            print(f"  {result}: {count}")

    # 保存 Q-learning 代理的 Q 表
    q_learning_agent.save_q_table("game_scores/cnn_cnnmcts/qlearning_agent_after_tournament.pkl")

    # 绘制胜率统计图
    plot_win_rate(results)

    print("\nTournament completed. Results and plots have been saved.")

