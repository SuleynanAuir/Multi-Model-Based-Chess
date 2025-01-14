import torch
import numpy as np
import chess

from cnn_self_play import apply_move, load_pytorch_model

# 游戏设置
num_games = 10  # 设定对弈的局数
##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 游戏结果统计
def play_multiple_games(model1, model2, num_games=100):
    # 记录每局游戏的胜负结果
    results = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}

    for game_num in range(num_games):
        print(f"Starting game {game_num + 1}...")
        
        # 进行一局游戏，记录胜负
        result = play_game_between_models(model1, model2)
        
        if result == 'model1':
            results['model1_wins'] += 1
        elif result == 'model2':
            results['model2_wins'] += 1
        else:
            results['draws'] += 1

    # 输出结果统计
    print("\nGame Results:")
    print(f"Model 1 Wins: {results['model1_wins']}")
    print(f"Model 2 Wins: {results['model2_wins']}")
    print(f"Draws: {results['draws']}")

    return results

# 与两个模型对弈的函数
def play_game_between_models(model1, model2):
    board_state = initial_board_state()  # 初始化棋盘状态
    move_number = 1
    is_model1_turn = True  # Model 1 starts first

    while not board_state.is_game_over():  # 判断游戏是否结束
        white_score, black_score = evaluate_board(board_state)  # 计算分数
        print_board_state(board_state, white_score, black_score, move_number)

        # 选择最佳棋步（Model 1 或 Model 2）
        if is_model1_turn:
            best_move = predict_best_move(model1, board_state)
        else:
            best_move = predict_best_move(model2, board_state)

        if best_move:
            board_state = apply_move(board_state, best_move)  # 应用最佳棋步
        else:
            print("No valid moves available.")
            break

        # 切换回合
        is_model1_turn = not is_model1_turn

        move_number += 1

    # 判断游戏结果
    result = board_state.result()  # 获取游戏结果
    if result == '1-0':  # 白方获胜
        return 'model1' if is_model1_turn else 'model2'
    elif result == '0-1':  # 黑方获胜
        return 'model2' if is_model1_turn else 'model1'
    else:  # 和局
        return 'draw'


# 初始化棋盘状态
def initial_board_state():
    board = chess.Board()  # 使用 python-chess 的 Board 类创建一个标准的棋盘
    return board

# 计算棋局分数（白方和黑方的分数）
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

# 获取合法的棋步
def get_legal_moves(board):
    legal_moves = [move.uci() for move in board.legal_moves]
    return legal_moves

# 预测最佳棋步
def predict_best_move(model, board_state):
    """
    预测棋步并选择最佳棋步
    """
    legal_moves = get_legal_moves(board_state)  # 获取所有合法棋步
    if not legal_moves:
        return None

    # 获取模型输入
    board_input = board_state_to_input(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 模型输出概率分布
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).squeeze().cpu().numpy()

    # 只选择合法棋步的概率
    legal_move_probs = {}
    for i, move in enumerate(legal_moves):
        legal_move_probs[move] = move_probs[i]

    # 根据概率分布选择最佳棋步
    best_move = max(legal_move_probs, key=legal_move_probs.get)
    return best_move

# 将棋盘状态转为模型输入格式（8x8x12）
def board_state_to_input(board):
    board_input = np.zeros((8, 8, 12))  # 8x8 的棋盘，12 个通道
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            piece_color = piece.color
            channel = piece_type - 1 + (6 if piece_color == chess.BLACK else 0)  # 0-5 是白棋，6-11 是黑棋
            row, col = divmod(square, 8)
            board_input[row, col, channel] = 1  # 标记该位置有棋子

    # 转换为 [batch_size, steps, channels, height, width]
    board_input = np.transpose(board_input, (2, 0, 1))  # 转换为 (channels, height, width)
    board_input = np.expand_dims(board_input, axis=0)  # 添加 batch_size 维度
    board_input = np.expand_dims(board_input, axis=1)  # 添加 steps 维度
    return board_input

# 游戏结束判断
def game_over(board):
    return board.is_game_over()

# 打印当前棋盘状态
def print_board_state(board, white_score, black_score, move_number):
    print(f"Move {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")

# 游戏运行
model1 = load_pytorch_model("chess_cnn_model.pth", num_classes=80)  # 假设 Model 1 路径
model2 = load_pytorch_model("mcts_cnn_model.pth", num_classes=80)  # 假设 Model 2 路径

# 启动多局对弈并记录结果
play_multiple_games(model1, model2, num_games=num_games)
