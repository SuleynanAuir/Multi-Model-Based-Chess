import torch
import numpy as np
import chess

from MCTS_module import MCTS_CNN_Model, load_model_with_missing_layers

# 加载 PyTorch 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('mcts_cnn_model_ql.pth', map_location=device,weights_only= True)
mcts_cnn_model = MCTS_CNN_Model(conv_size=32, conv_depth=3, num_classes=64).to(device)
mcts_cnn_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# 加载模型时自动处理缺失层
load_model_with_missing_layers(mcts_cnn_model, checkpoint)
model = mcts_cnn_model

# 将模型设置为评估模式
mcts_cnn_model.eval()

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



# 初始化棋盘状态
def initial_board_state():
    board = chess.Board()  # 使用 python-chess 的 Board 类创建一个标准的棋盘
    return board

# 游戏结束判断函数
def game_over(board):
    return board.is_game_over()  # 判断棋局是否结束

# 获取合法的棋步
def get_legal_moves(board):
    legal_moves = [move.uci() for move in board.legal_moves]
    return legal_moves

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


# 模拟游戏的棋步应用
def apply_move(board, move):
    """
    应用给定的棋步并更新棋盘状态。
    """
    chess_move = chess.Move.from_uci(move)  # 将字符串格式的棋步转为棋盘上的 Move 对象
    if chess_move in board.legal_moves:
        board.push(chess_move)  # 执行这个棋步
    else:
        print("Invalid move:", move)
    return board

# 计算棋局分数（白方和黑方的分数）
def evaluate_board(board):
    """
    根据棋盘上的棋子数量计算一个评分，分别计算白方和黑方的分数。
    假设棋子的分数：兵 = 1，马和象 = 3，车 = 5，后 = 9
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

# 打印当前棋盘状态
def print_board_state(board, white_score, black_score, move_number):
    print(f"Move {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")

# 写入棋盘和分数到文件
def log_game_state(board, white_score, black_score, move_number, file_path="game_log1.txt"):
    """
    将棋盘状态和分数记录到文件中
    """
    with open(file_path, "a") as file:
        file.write(f"Move {move_number}:\n")
        file.write(str(board) + "\n")
        file.write(f"White score: {white_score}\n")
        file.write(f"Black score: {black_score}\n")
        file.write("-" * 40 + "\n")

# 主函数：与模型对弈
def play_game_M2M(model):
    board_state = initial_board_state()  # 初始化棋盘状态
    move_history = []  # 用于存储合法的棋步
    
    """
    这些参数都是可以改的
    """
    
    max_moves = 100  # 积累合法棋步的数量（100）
    move_number = 1  # 当前步数

    while not game_over(board_state):  # 判断游戏是否结束
        white_score, black_score = evaluate_board(board_state)  # 计算白方和黑方的分数
        print_board_state(board_state, white_score, black_score, move_number)  # 打印棋盘状态和分数

        # 积累合法的棋步
        legal_moves = get_legal_moves(board_state)
        move_history.extend(legal_moves)

        # 如果积累了足够的合法棋步，选择最佳棋步
        if len(move_history) >= max_moves:
            best_move = predict_best_move(model, board_state)
            print(f"Best move: {best_move}")
            board_state = apply_move(board_state, best_move)  # 应用最佳棋步
            move_history = []  # 清空积累的棋步列表
        else:
            # 如果还没有积累足够的棋步，可以采取其他策略（如随机选择合法棋步）
            move = np.random.choice(legal_moves)  # 随机选择一个棋步
            print(f"Random move: {move}")
            board_state = apply_move(board_state, move)

        # 记录当前棋局状态和分数到文件
        log_game_state(board_state, white_score, black_score, move_number)  # 记录棋局状态和分数到文件

        # 增加步数
        move_number += 1

    # 主函数：与模型对弈
def play_game_M2M(model):
    board_state = initial_board_state()
    move_number = 1

    while not board_state.is_game_over():  # 判断游戏是否结束
        white_score, black_score = evaluate_board(board_state)  # 计算分数
        print_board_state(board_state, white_score, black_score, move_number)

        # 选择最佳棋步
        best_move = predict_best_move(model, board_state)
        if best_move:
            print(f"Best move: {best_move}")
            # print(f"")
            board_state = apply_move(board_state, best_move)
        else:
            print("No valid moves available.")
            break

        # 记录当前棋局状态和分数
        log_game_state(board_state, white_score, black_score, move_number)

        move_number += 1

    print("Game over!")
    print(game_over_reason(board_state), '\n')
    print_winner(board_state)



def print_board_state(board, white_score, black_score, move_number):
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)

# 判断游戏结束的原因
def game_over_reason(board_state):
    if board_state.is_checkmate():
        return "Checkmate! Game Over."
    elif board_state.is_stalemate():
        return "Stalemate! Game Over."
    elif board_state.is_insufficient_material():
        return "Insufficient Material! Game Over."
    elif board_state.is_seventyfive_moves():
        return "Seventy-five moves rule! Game Over."
    elif board_state.is_fivefold_repetition():
        return "Fivefold repetition rule! Game Over."
    elif board_state.is_variant_draw():
        return "Variant draw! Game Over."
    else:
        return "Game Over!"


# 打印获胜方
def print_winner(board_state):
    result = board_state.result()  # 获取游戏结果
    if result == '1-0':
        print("White wins!")
    elif result == '0-1':
        print("Black wins!")
    elif result == '1/2-1/2':
        print("It's a draw!")

     #判断游戏结束的原因


if __name__ == "__main__":
    # 启动与模型对弈
    play_game_M2M(model)