import os
import time
import chess
from matplotlib import pyplot as plt

import matplotlib
import torch
import numpy as np

from M2M import get_legal_moves
from mcts import mcts_pred, node
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 模型 2 的输入格式
def board_state_to_input_model2(board):
    board_input = np.zeros((8, 8, 12))  # 8x8 board with 12 channels (one for each piece type)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            channel = piece_type - 1 + (6 if piece_color == chess.BLACK else 0)  # 0-5 for white, 6-11 for black
            row, col = divmod(square, 8)
            board_input[row, col, channel] = 1  # Mark the position with a piece
    return board_input


# 加载模型 2
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
    checkpoint = torch.load(file_path, map_location=device,weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# 模型 1 的预测最佳棋步
def predict_best_move_model1(board_state):
    root = node()
    root.state = board_state
    white = 1 if board_state.turn == chess.WHITE else 0  # 判断当前是否轮到白棋

    # 获取最佳棋步（algebraic notation）
    best_move_san = mcts_pred(root, board_state.is_game_over(), white, iterations=10)
    if best_move_san is None:
        return None

    # 转换为 UCI 格式
    try:
        best_move = board_state.parse_san(best_move_san).uci()
    except ValueError as e:
        print(f"Error converting SAN to UCI: {e}")
        return None

    return best_move

# 模型 2 的预测最佳棋步
def predict_best_move_model2(model, board_state):
    model = model.to(device) 
    legal_moves = get_legal_moves(board_state)

    if not legal_moves:
        return None  # Return None if there are no legal moves

    best_move = None
    best_move_value = -float('inf')  # Store the value of the best move

    # Accumulate legal moves and compute the predicted value for each move
    for move in legal_moves:
        # Convert board state to model input
        board_input = board_state_to_input_model2(board_state)
        board_input = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)  # Convert to PyTorch tensor

        # Get model prediction
        with torch.no_grad():
            prediction = model(board_input)  # Get prediction

        # Assume the model output is a tensor with 64 values, each representing a move's predicted value
        move_index = legal_moves.index(move)
        move_value = prediction[0, move_index].item()  # Get the predicted value for the current move

        # Update the best move if the current move has a higher value
        if move_value > best_move_value:
            best_move_value = move_value
            best_move = move

    return best_move

# 应用棋步
def apply_move(board, move):
    chess_move = chess.Move.from_uci(move)
    if chess_move in board.legal_moves:
        board.push(chess_move)


def print_board_state(board, white_score, black_score, move_number):
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)
    # time.sleep(0.0000001)  # Add a 1-second delay


def evaluate_board(board):
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

def log_game_state(board, white_score, black_score, move_number, file_path="game_log4.txt"):
    with open(file_path, "a") as file:
        file.write(f"Move {move_number}:\n")
        file.write(str(board) + "\n")
        file.write(f"White score: {white_score}\n")
        file.write(f"Black score: {black_score}\n")
        file.write("-" * 4 + "\n")
# 游戏结束判断
def game_over(board):
    return board.is_game_over()

# 游戏对弈逻辑
def play_game(algorithm1, algorithm2 ,game_number, save_dir = "game_scores/mcts_cnn"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    board = chess.Board()
    current_player = 1  # 1 表示 MCTS 算法，2 表示 CNN 算法
    move_number = 1
    white_scores = []
    black_scores = []
    while not game_over(board):
        white_score, black_score = evaluate_board(board)
        print_board_state(board, white_score, black_score, move_number)
        white_scores.append(white_score)
        black_scores.append(black_score)
        if current_player == 1:#white
            # 调用 MCTS 算法预测最佳棋步
            move = predict_best_move_model1(board)
        else:#black
            # 调用 CNN 算法预测最佳棋步
            move = predict_best_move_model2(algorithm2, board)

        if move is None:
            break  # 如果没有合法棋步，直接结束对局

        apply_move(board, move)  # 应用棋步
        current_player = 3 - current_player  # 切换玩家
        log_game_state(board, white_score, black_score, move_number)  # 记录对局状态

        move_number += 1
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(white_scores) + 1), white_scores, label="mcts", marker='o')
    plt.plot(range(1, len(black_scores) + 1), black_scores, label="cnn", marker='x')
    plt.title(f"Game {game_number} Score Progression")
    plt.xlabel("Move Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # 保存图表到指定文件夹
    plot_path = os.path.join(save_dir, f"game_{game_number}_scores.png")
    plt.savefig(plot_path)
    plt.close()  
    return board.result() 

# 多局对弈统计
def run_tournament(algorithm1, algorithm2, games=10):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for game in range(games):
        result = play_game(algorithm1, algorithm2,game_number=game)
        results[result] += 1
        print(f"Game {game + 1}: Result = {result}")
    return results

def plot_win_rate(results):
    # 提取胜、负、平局的数量
    white_wins = results["1-0"]
    black_wins = results["0-1"]
    draws = results["1/2-1/2"]
    
    # 数据
    labels = ['White Wins', 'Black Wins', 'Draws']
    win_counts = [white_wins, black_wins, draws]
    
    # 绘制条形图
    fig, ax = plt.subplots()
    ax.bar(labels, win_counts, color=['green', 'red', 'blue'])
    
    # 添加标题和标签
    ax.set_title('Model Tournament Results')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Number of Games')
    
    # 显示胜率
    total_games = sum(win_counts)
    win_rates = [round((count / total_games) * 100, 2) for count in win_counts]
    
    for i, win_rate in enumerate(win_rates):
        ax.text(i, win_counts[i] + 0.1, f'{win_rate}%', ha='center', va='bottom')
    
    # 显示图形
    plot_path = os.path.join("game_scores/mcts_cnn", "tournament_results.png")
    plt.savefig(plot_path)
    plt.close(fig)





# 加载模型
# algorithm1, device1 = load_algorithm1_model("mcts_cnn_model_ql.pth", num_classes=64)
algorithm1 = 0
algorithm2 = load_algorithm2_model("chess_cnn_model_ql.pth", num_classes=80)

# 开始比赛
results = run_tournament(algorithm1, algorithm2,games=10)

# 输出最终结果
print("\nFinal Results:", results)
plot_win_rate(results)