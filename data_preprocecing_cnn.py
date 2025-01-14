import zstandard as zstd
import chess.pgn
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataPreprocessing:
    def __init__(self):
        self.input_filename = 'Data_Base/lichess_db_standard_rated_2013-01.pgn.zst'
        self.output_filename = 'output_pgn_file.pgn'
        self.csv_filename = 'chess_games.csv'
        self.cnn_output_dir = 'cnn_batches'
        self.batch_size = 1000
        self.df = None
        self.input_data = []
        self.each_game_steps_num = []
        self.best_upper = None

    def decompress_zst(self):
        with tqdm(total=1, desc="Decompressing ZST file", unit="file") as pbar:
            with open(self.input_filename, 'rb') as f_in:
                with open(self.output_filename, 'wb') as f_out:
                    dctx = zstd.ZstdDecompressor()
                    dctx.copy_stream(f_in, f_out)
                    pbar.update(1)

    def extract_pgn_data(self):
        chunk_size=20000
        games = []
        with tqdm(total=1, desc="Extracting PGN data", unit="games") as pbar:
            with open(self.output_filename) as f:
                game_count = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    headers = game.headers
                    white_player = headers.get("White", "Unknown")
                    black_player = headers.get("Black", "Unknown")
                    result = headers.get("Result", "*")
                    date = headers.get("Date", "?")
                    moves = []
                    board = game.board()
                    for move in game.mainline_moves():
                        moves.append(str(move))
                    games.append({
                        "White": white_player,
                        "Black": black_player,
                        "Result": result,
                        "Date": date,
                        "Moves": " ".join(moves)
                    })
                    game_count += 1

                    # 更新进度条显示已处理的局数
                    pbar.update(1)

                    # 每处理 chunk_size 局之后写入 CSV 文件
                    if game_count % chunk_size == 0:
                        df_chunk = pd.DataFrame(games)
                        write_mode = 'a' if os.path.exists(self.csv_filename) else 'w'
                        df_chunk.to_csv(self.csv_filename, mode=write_mode, header=write_mode == 'w', index=False)
                        games = []  # 清空缓存

                # 处理剩余的局数
                if games:
                    df_chunk = pd.DataFrame(games)
                    write_mode = 'a' if os.path.exists(self.csv_filename) else 'w'
                    df_chunk.to_csv(self.csv_filename, mode=write_mode, header=write_mode == 'w', index=False)


    def preprocess_csv(self):
        with tqdm(total=1, desc="Preprocessing CSV", unit="file") as pbar:
            self.df = pd.read_csv(self.csv_filename)

            # 去除 `Moves` 列为空的行
            self.df = self.df.dropna(subset=['Moves'])

            # 去除 `Date` 列
            if 'Date' in self.df.columns:
                self.df = self.df.drop(columns=['Date'])

            # 保存处理后的 CSV 文件
            self.df.to_csv(self.csv_filename, index=False)
            pbar.update(1)

    def fen_to_matrix(self, fen):
        piece_map = {'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
                     'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6, '.': 0}
        board_matrix = []
        fen_part = fen.split()[0]
        for row in fen_part.split('/'):
            for char in row:
                if char.isdigit():
                    board_matrix.extend([0] * int(char))
                else:
                    board_matrix.append(piece_map.get(char, 0))
        return board_matrix


    def convert_move_to_matrix(self, moves):
        matrices = []
        board = chess.Board()
        for move in moves.split():
            board.push(chess.Move.from_uci(move))
            board_matrix = self.fen_to_matrix(board.fen())
            matrices.append(board_matrix)
        return matrices

    def calculate_game_steps(self, chunksize=5000, max_workers=40):
        self.df = pd.read_csv(self.csv_filename)
        all_moves = []
        with tqdm(total=1, desc="Loading CSV in chunks", unit="chunk") as pbar:
            for chunk in pd.read_csv(self.csv_filename, chunksize=chunksize):
                all_moves.extend(chunk['Moves'].dropna().tolist())
                pbar.update(1)

        num_games = len(all_moves)
        start_points = [0, num_games // 4, num_games // 2, 3 * num_games // 4]
        results = [[] for _ in range(4)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            with tqdm(total=num_games, desc="Calculating game steps", unit="game") as pbar:
                for idx, start in enumerate(start_points):
                    end = start + (num_games // 4) if idx < 3 else num_games
                    futures.append(executor.submit(self.process_game_steps, all_moves[start:end], results[idx]))

                for future in as_completed(futures):
                    future.result()
                    pbar.update(num_games // 4)

        self.each_game_steps_num = [step for result in results for step in result]
        self.best_upper = math.ceil(self.get_iqr_based_range(self.each_game_steps_num)[1])

    def process_game_steps(self, moves, result_list):
        for move_sequence in moves:
            single_board_matrices = self.convert_move_to_matrix(move_sequence)
            result_list.append(len(single_board_matrices))

    def get_iqr_based_range(self, my_list):
        Q1 = np.percentile(my_list, 25)
        Q3 = np.percentile(my_list, 75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    def limit_steps(self, game, max_step):
        steps = len(game)
        if steps < max_step:
            padding = np.zeros((max_step - steps, 64), dtype = np.int8)
            game = np.vstack([game, padding],dtype= np.int8)
        elif steps > max_step:
            game = game[:max_step]
        return game

    def reshape_games(self):
        all_moves = self.df['Moves']
        num_games = len(all_moves)
        memmap_filename = 'reshaped_games_memmap.dat'
        reshaped_games_memmap = np.memmap(memmap_filename, dtype=np.int8, mode='w+', shape=(num_games, self.best_upper, 64))

        with tqdm(total=num_games, desc="Reshaping games", unit="game") as pbar:
            for i in range(num_games):
                single_board_matrices = self.convert_move_to_matrix(all_moves[i])
                reshaped_game = self.limit_steps(single_board_matrices, max_step=self.best_upper)
                reshaped_games_memmap[i] = np.array(reshaped_game, dtype=np.int8)  # Store directly into memmap to reduce memory usage
                pbar.update(1)

        self.input_data = reshaped_games_memmap

    def prepare_cnn_input(self):
        if not os.path.exists(self.cnn_output_dir):
            os.makedirs(self.cnn_output_dir)
        buffer_matrices = []
        with tqdm(total=len(self.input_data), desc="Preparing CNN input", unit="game") as pbar:
            for batch_idx in range(0, len(self.input_data), self.batch_size):
                batch_data = self.input_data[batch_idx: batch_idx + self.batch_size]
                all_matrices = list()
                for each_game in batch_data:
                    board_matrices = each_game
                    matrices_3d = np.zeros((len(board_matrices), 8, 8, 12), dtype=np.int8)
                    for i, board_matrix in enumerate(board_matrices):
                        for j, piece_value in enumerate(board_matrix):
                            row = j // 8
                            col = j % 8
                            matrices_3d[i, row, col, int(piece_value)] = piece_value
                    all_matrices.append(matrices_3d)
                buffer_matrices.extend(all_matrices)
                if len(buffer_matrices) >= self.batch_size or batch_idx + self.batch_size >= len(self.input_data):
                    batch_output = np.array(buffer_matrices)
                    buffer_matrices = []
                    filename = os.path.join(self.cnn_output_dir, f"batch_{batch_idx}.npy")
                    np.save(filename, batch_output)
                pbar.update(len(batch_data))

    def run(self):
        self.input_filename = 'lichess_db_standard_rated_2015-12.pgn.zst'
        self.output_filename = 'output_pgn_file_400w.pgn'
        self.csv_filename = 'chess_games.csv'
        self.cnn_output_dir = 'cnn_batches'
        self.batch_size = 25000
        
        self.decompress_zst()
        self.extract_pgn_data()
        self.preprocess_csv()
        self.calculate_game_steps()
        self.reshape_games()
        self.prepare_cnn_input()


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing()
    data_preprocessing.run()
