import numpy as np
import pandas as pd
import chess
import os
from tqdm import tqdm

class ChessDataPreprocessor:
    def __init__(self, csv_file, max_steps):
        self.data = pd.read_csv(csv_file)
        self.max_moves = max_steps
        self.batch_size = 25000
        self.output_dir = "output_mcts_cnn"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def split_dims(self, board):
        """将棋盘状态转换为 (12, 8, 8) 的张量"""
        board3d = np.zeros((12, 8, 8), dtype=np.int8)
        for piece in chess.PIECE_TYPES:
            # 白方棋子
            for square in board.pieces(piece, chess.WHITE):
                row, col = divmod(square, 8)
                board3d[piece - 1, row, col] = 1
            # 黑方棋子
            for square in board.pieces(piece, chess.BLACK):
                row, col = divmod(square, 8)
                board3d[5 + piece, row, col] = 1
        return board3d

    def move_to_integer(self, move, board):
        """将棋步转换为合法棋步的索引"""
        move_obj = chess.Move.from_uci(move)
        legal_moves = list(board.legal_moves)
        try:
            move_index = next(i for i, m in enumerate(legal_moves) if m == move_obj)
        except StopIteration:
            move_index = 0  # 使用 0 表示非法棋步
        return move_index

    def limit_steps(self, game):
        """限制棋局步数，填充不足或截断多余的步数"""
        steps = len(game)
        if steps < self.max_moves:
            padding = np.zeros((self.max_moves - steps, 12, 8, 8), dtype=np.int8)
            game = np.vstack([game, padding])
        elif steps > self.max_moves:
            game = game[:self.max_moves]
        return game

    def limit_labels(self, labels):
        """限制标签步数，填充不足或截断多余的步数"""
        steps = len(labels)
        if steps < self.max_moves:
            padding = np.zeros(self.max_moves - steps, dtype=np.int8)
            labels = np.concatenate([labels, padding])
        elif steps > self.max_moves:
            labels = labels[:self.max_moves]
        return labels

    def find_optimal_max_steps(self):
        min_steps, max_steps = float('inf'), 0
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Finding optimal max_steps"):
            game_moves = row['Moves']
            board = chess.Board()
            steps = 0

            for move in game_moves.split():
                steps += 1
                board.push(chess.Move.from_uci(move))

            min_steps = min(min_steps, steps)
            max_steps = max(max_steps, steps)

        optimal_max_steps = max_steps
        print(f"Optimal max_steps value: {optimal_max_steps}")
        return optimal_max_steps

    def prepare_and_save_data_in_batches(self):
        all_boards, labels = [], []
        batch_counter = 0

        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Preparing data in batches"):
            game_moves = row['Moves']
            board = chess.Board()
            boards, game_labels = [], []

            for move in game_moves.split():
                board_state = self.split_dims(board)
                boards.append(board_state)
                move_index = self.move_to_integer(move, board)
                game_labels.append(max(0, move_index))
                board.push(chess.Move.from_uci(move))

            while len(boards) < self.max_moves:
                boards.append(np.zeros((12, 8, 8), dtype=np.int8))
                game_labels.append(0)

            boards = self.limit_steps(np.array(boards))
            game_labels = self.limit_labels(np.array(game_labels).reshape(-1, 1)).reshape(-1)

            all_boards.append(boards)
            labels.append(game_labels)

            if (idx + 1) % self.batch_size == 0 or (idx + 1) == len(self.data):
                all_boards = np.array(all_boards, dtype=np.int8)
                labels = np.array(labels, dtype=np.int8)
                labels = np.clip(labels, 0, 63)

                batch_file_path = os.path.join(self.output_dir, f"batch_{batch_counter}.npz")
                np.savez(batch_file_path, boards=all_boards, labels=labels)
                print(f"Saved batch {batch_counter} to {batch_file_path}")

                batch_counter += 1
                all_boards, labels = [], []
