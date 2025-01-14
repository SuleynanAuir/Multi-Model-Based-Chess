import os
import chess
import chess.pgn
import chess.engine
import random
import time
import heapq
from math import log, sqrt, e, inf

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
# ===========================
# 部分一：定义 Minimax AI
# ===========================

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
            max_eval = -inf
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = inf
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval)
            return min_eval

    def get_best_move(self, board):
        best_move = None
        best_value = -inf
        for move in board.legal_moves:
            board.push(move)
            move_value = self.minimax(board, self.depth - 1, False)
            board.pop()
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move

# ===========================
# 部分二：定义 MCTS AI
# ===========================

# 初始化 Stockfish 引擎
engine_path = r"stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
try:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
except Exception as e:
    print(f"无法启动 Stockfish 引擎: {e}")
    engine = None

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # chess.Board 对象
        self.parent = parent  # 父节点
        self.move = move  # 导致当前状态的走法 (chess.Move)
        self.children = set()  # 子节点集合
        self.N = 0  # 节点被访问的次数
        self.n = 0  # 节点被选中的次数
        self.v = 0  # 节点的价值
        self.ucb = 0  # UCB 值

    def __lt__(self, other):
        return self.ucb < other.ucb

def ucb1(curr_node):
    if curr_node.n == 0:
        return inf  # 确保未被访问过的节点优先被选中
    return curr_node.v / curr_node.n + 2 * sqrt(log(curr_node.parent.N + 1) / curr_node.n)

def rollout(curr_node):
    if curr_node.state.is_game_over():
        result = curr_node.state.result()
        if result == '1-0':
            return (1, curr_node)
        elif result == '0-1':
            return (-1, curr_node)
        else:
            return (0.5, curr_node)
    # 使用 Stockfish 进行快照
    try:
        if engine:
            result = engine.play(curr_node.state, chess.engine.Limit(time=0.00001))
            move = result.move
        else:
            # 如果 Stockfish 未启动，则随机走
            move = random.choice(list(curr_node.state.legal_moves))
    except Exception as e:
        print(f"Stockfish 选择走法时出错: {e}. 使用随机走法。")
        move = random.choice(list(curr_node.state.legal_moves))
    
    tmp_state = curr_node.state.copy()
    tmp_state.push(move)
    
    # 创建新的子节点
    child_node = MCTSNode(tmp_state, parent=curr_node, move=move)
    curr_node.children.add(child_node)
    return rollout(child_node)

def expand(curr_node):
    if not curr_node.children:
        # 创建所有可能的子节点
        for move in curr_node.state.legal_moves:
            tmp_state = curr_node.state.copy()
            tmp_state.push(move)
            child_node = MCTSNode(tmp_state, parent=curr_node, move=move)
            curr_node.children.add(child_node)
    # 选择 UCB1 值最高的子节点
    selected_child = max(curr_node.children, key=lambda node: ucb1(node))
    return selected_child

def mcts_pred(root, iterations=100):
    for _ in range(iterations):
        node = root
        # Selection and Expansion
        while node.children:
            node = expand(node)
        # Simulation
        reward, _ = rollout(node)
        # Backpropagation
        while node is not None:
            node.N += 1
            node.n += 1
            node.v += reward
            node = node.parent
    # 选择访问次数最多的子节点
    if not root.children:
        return None
    best_child = max(root.children, key=lambda node: node.N)
    return best_child.move

def mcts_ai(board, iterations=100):
    if engine is None:
        print("Stockfish 引擎未启动，MCTS AI 无法运行。")
        return None
    root = MCTSNode(board.copy())
    move = mcts_pred(root, iterations)
    return move

# ===========================
# 部分三：定义对弈逻辑
# ===========================

def apply_move(board, move):
    if isinstance(move, chess.Move) and move in board.legal_moves:
        board.push(move)
    else:
        print(f"非法走法尝试：{move} (类型: {type(move)})")

def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
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

def play_game(minimax_ai, mcts_ai,game_number, save_dir="game_scores/minimax_mcts" ,log=True):
    board = chess.Board()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    move_number = 1
    results = []
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Minimax vs MCTS"
    pgn_game.headers["Result"] = "*"

    node_pgn = pgn_game
    white_scores = []
    black_scores = []
    while not game_over(board):
        white_score, black_score = evaluate_board(board)
        print_board_state(board, white_score, black_score, move_number)
        white_scores.append(white_score)
        black_scores.append(black_score)
        if board.turn == chess.WHITE:
            move = minimax_ai.get_best_move(board)
            player = "Minimax"
        else:
            move = mcts_ai(board, iterations=100)
            player = "MCTS"

        # Debug: Check move type
        if not isinstance(move, chess.Move):
            print(f"Error: {player} 返回了无效的走法类型: {type(move)}。Move: {move}")
            break

        if move not in board.legal_moves:
            print(f"Error: {player} 返回了非法走法: {move}")
            break

        # **在应用走法之前获取 SAN 表示**
        if log:
            try:
                san_move = board.san(move)
                print(f"Move {move_number}: {player} plays {san_move} (type: {type(move)})")
                node_pgn = node_pgn.add_variation(move)
            except ValueError as e:
                print(f"Move {move_number}: {player} plays {move} (非法走法) - {e}")

        # Apply move
        apply_move(board, move)
        results.append(move.uci())

        log_game_state(board, white_score, black_score, move_number)

        move_number += 1

    result = board.result()
    pgn_game.headers["Result"] = result

    if log:
        print(board)
        print(f"Game Result: {result}")
        print("Moves:", " ".join(results))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(white_scores) + 1), white_scores, label="minimax", marker='o')
    ax.plot(range(1, len(black_scores) + 1), black_scores, label="mcts", marker='x')
    ax.set_title(f"Game {game_number} Score Progression")
    ax.set_xlabel("Move Number")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    # 保存图表到指定文件夹
    plot_path = os.path.join(save_dir, f"game_{game_number}_scores.png")
    plt.savefig(plot_path)
    plt.close(fig)
    return result, pgn_game

def run_tournament(minimax_ai, mcts_ai, games=10):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    for game in range(1, games + 1):
        print(f"\nStarting Game {game}")
        result, pgn_game = play_game(minimax_ai, mcts_ai,game_number=game)
        results[result] += 1
        # 可以将 PGN 保存到文件或其他地方
    return results

# ===========================
# 部分四：主函数
# ===========================
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
    plot_path = os.path.join("game_scores/minimax_mcts", "tournament_results.png")
    plt.savefig(plot_path)
    plt.close(fig)

def main():
    # 初始化 AI
    minimax_ai = MinimaxAI(depth=3)
    # MCTS 不需要初始化

    # 运行比赛
    num_games = 10  # 设置您想要进行的对弈局数
    results = run_tournament(minimax_ai, mcts_ai, games=num_games)

    # 输出最终结果
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    plot_win_rate(results)
    # 关闭 Stockfish 引擎
    if engine is not None:
        engine.quit()

if __name__ == "__main__":
    main()

