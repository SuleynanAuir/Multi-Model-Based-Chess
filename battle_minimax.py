import os
import random
from matplotlib import pyplot as plt
import pygame as p
import math
import copy
import time
import chess
import torch
import numpy as np

class Chess_Game_State():
    def __init__(self):
        self.board = chess.Board()
        self.moveLog = []
        self.white_turn = True
        self.check_mate = False
        self.stale_mate = False
        self.is_enpassant = ()
        self.current_castle_right = self.board.castling_rights  # 使用 python-chess 的 castling_rights

    def chess_move(self, move):
        # move 是 UCI 格式的字符串
        chess_move = chess.Move.from_uci(move)
        if chess_move in self.board.legal_moves:
            self.board.push(chess_move)
            self.moveLog.append(move)
            self.white_turn = not self.white_turn
            self.check_mate = self.board.is_checkmate()
            self.stale_mate = self.board.is_stalemate()
        else:
            print(f"非法走法：{move}")

    def undo_move(self):
        if self.moveLog:
            self.board.pop()
            self.moveLog.pop()
            self.white_turn = not self.white_turn
            self.check_mate = self.board.is_checkmate()
            self.stale_mate = self.board.is_stalemate()

    def get_valid_move(self):
        return list(self.board.legal_moves)

    def getbest_move(self, depth, is_max):
        # 简化版的 Minimax 实现，只考虑下一步
        best_move = None
        best_value = -math.inf if is_max else math.inf
        for move in self.get_valid_move():
            self.board.push(move)
            board_value = self.evaluate_board()
            self.board.pop()
            if is_max:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move
        return best_move.uci() if best_move else None

    def evaluate_board(self):
        # 简单的评估函数
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        value = 0
        for piece_type in piece_values:
            value += len(self.board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            value -= len(self.board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
        return value


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_algorithm1_model(file_path, num_classes=64):
    from MCTS_module import MCTS_CNN_Model, load_model_with_missing_layers
    model = MCTS_CNN_Model(conv_size=32, conv_depth=3, num_classes=num_classes).to(device)
    checkpoint = torch.load(file_path, map_location=device)
    load_model_with_missing_layers(model, checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

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
    board_input = np.expand_dims(board_input, axis=1)
    return board_input

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
    return board_input
def get_legal_moves(board):
    return [move.uci() for move in board.legal_moves]

def predict_best_move_model1(model, board_state):
    legal_moves = get_legal_moves(board_state)
    if not legal_moves:
        return None

    board_input = board_state_to_input_model1(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).squeeze().cpu().numpy()

    legal_move_probs = {}
    for i, move in enumerate(legal_moves):
        legal_move_probs[move] = move_probs[i]

    best_move = max(legal_move_probs, key=legal_move_probs.get)
    return best_move

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

def apply_move(board, move):
    chess_move = chess.Move.from_uci(move)
    if chess_move in board.legal_moves:
        board.push(chess_move)
    else:
        print(f"非法走法：{move}")

def evaluate_board(board):
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
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)

def game_over(board):
    return board.is_game_over()

def log_game_state(board, white_score, black_score, move_number, file_path="game_log.txt"):
    with open(file_path, "a") as file:
        file.write(f"Move {move_number}:\n")
        file.write(str(board) + "\n")
        file.write(f"White score: {white_score}\n")
        file.write(f"Black score: {black_score}\n")
        file.write("-" * 18 + "\n")

def play_game(algorithm1, algorithm2,game_number,algorithm_name,save_dir="game_scores/minimax"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    board = chess.Board()
    gs = Chess_Game_State()
    current_player = 1
    move_number = 1
    white_scores = []
    black_scores = []
    while not game_over(board):
        white_score, black_score = evaluate_board(board)
        white_scores.append(white_score)
        black_scores.append(black_score)
        print_board_state(board, white_score, black_score, move_number)
        if current_player == 1:
            move = algorithm1(gs, board)
        else:
            move = algorithm2(board)

        print(f"Legal Moves: {get_legal_moves(board)}")
        print(f"Predicted Move: {move}")
        if move is None:
            break
        apply_move(board, move)
        gs.chess_move(move)  # 同步自定义棋盘状态

        current_player = 3 - current_player
        log_game_state(board, white_score, black_score, move_number)

        move_number += 1
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(white_scores) + 1), white_scores, label="minimax", marker='o')
    plt.plot(range(1, len(black_scores) + 1), black_scores, label=algorithm_name, marker='x')
    plt.title(f"Game {game_number} Score Progression")
    plt.xlabel("Move Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # 保存图表到指定文件夹
    plot_path = os.path.join(save_dir, f"game_{game_number}_scores_{algorithm_name}.png")
    plt.savefig(plot_path)
    plt.close()
    return board.result()

def run_tournament(algorithm1, algorithm2, algorithm_name,games=10):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for game in range(games):
        print(f"\nStarting Game {game + 1}")
        result = play_game(algorithm1, algorithm2,game_number=game,algorithm_name=algorithm_name)
        results[result] += 1
        print(f"Game {game + 1}: Result = {result}")
    return results

def plot_win_rate(results,algorithm_name):
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
    plot_path = os.path.join("game_scores/minimax", f"tournament_results_{algorithm_name}.png")
    plt.savefig(plot_path)
    plt.close(fig)
# 加载模型
algorithm1_model = load_algorithm1_model("mcts_cnn_model.pth", num_classes=64)
algorithm2_model = load_algorithm2_model("chess_cnn_model.pth", num_classes=80)

# 定义算法函数
def algorithm1(gs, board):
    return gs.getbest_move(depth=2, is_max=gs.white_turn)

def algorithm2(board):
    return predict_best_move_model1(algorithm1_model, board)

def algorithm3(board):
    return predict_best_move_model2(algorithm2_model, board)
def main():
    # 开始比赛，自定义 AI 对阵模型 1
    results1 = run_tournament(algorithm1, algorithm2, "cnn_mcts",games=100)


    # 开始比赛，自定义 AI 对阵模型 2
    results2 = run_tournament(algorithm1, algorithm3,"cnn", games=100)
    print("\nFinal Results (Custom AI vs Model 2):", results1)
    print("\nFinal Results (Custom AI vs Model 1):", results2)
    plot_win_rate(results1,"cnn")
    plot_win_rate(results2,"cnn_mcts")
if __name__ == "__main__":
    main()