import time
import os
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from battle import (
    board_state_to_input_model1,
    board_state_to_input_model2,
    load_algorithm1_model,
    load_algorithm2_model,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau  # 使用 ReduceLROnPlateau

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载对手模型
# 请确保 "mcts_cnn_model.pth" 和 "chess_cnn_model_ql.pth" 存在且可用
algorithm1, device1 = load_algorithm1_model("mcts_cnn_model.pth", num_classes=64)
algorithm2 = load_algorithm2_model("chess_cnn_model_ql.pth", num_classes=80)

ACTION_SIZE = 64 * 64

def get_legal_moves(board):
    # board应为chess.Board对象
    return [move for move in board.legal_moves]

def predict_best_move_model1(model, board_state):
    """
    预测最佳移动（模型1）
    确保输入张量具有5个维度：[batch_size, steps, channels, height, width]
    """
    legal_moves = list(get_legal_moves(board_state))
    if not legal_moves:
        return None
    board_input = board_state_to_input_model1(board_state)
    
    # 确保board_input是numpy数组
    if not isinstance(board_input, np.ndarray):
        board_input = np.array(board_input)
    
    # 移除任何单例维度
    board_input = np.squeeze(board_input)
    
    # 如果仍为3D，添加批次维度
    if board_input.ndim == 3:
        board_input = np.expand_dims(board_input, axis=0)
    
    # 转置为 [batch_size, channels, height, width]
    if board_input.shape[-1] in [12, 64, 80]:  # 根据实际通道数调整
        board_input = np.transpose(board_input, (0, 3, 1, 2))
    
    # 添加 steps 维度，假设 steps=1
    board_input = np.expand_dims(board_input, axis=1)  # [batch_size, steps, channels, height, width]
    
    # 确保输入为5D
    if board_input.ndim != 5:
        raise ValueError(f"期望 board_input 具有5个维度，但得到 {board_input.ndim}")
    
    # 转换为torch张量
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).squeeze().cpu().numpy()
    
    if len(move_probs) < len(legal_moves):
        move_probs = move_probs[:len(legal_moves)]
    move_dict = {m.uci(): p for m, p in zip(legal_moves, move_probs)}
    best_move = max(move_dict, key=move_dict.get)
    return chess.Move.from_uci(best_move)

def predict_best_move_model2(model, board_state):
    # board_state是chess.Board对象
    legal_moves = list(get_legal_moves(board_state))
    if not legal_moves:
        return None
    board_input = board_state_to_input_model2(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(board_input).squeeze().cpu().numpy()
    if len(prediction) < len(legal_moves):
        prediction = prediction[:len(legal_moves)]
    move_values = {m.uci(): v for m, v in zip(legal_moves, prediction)}
    best_move = max(move_values, key=move_values.get)
    return chess.Move.from_uci(best_move)

def predict_best_move_dqn(model, board_state, epsilon=0.1):
    # board_state是chess.Board对象
    legal_moves = list(get_legal_moves(board_state))
    if not legal_moves:
        return None
    board_input = board_state_to_rl_input(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        q_values = model(board_input).squeeze().cpu().numpy()
    legal_actions = []
    for move in board_state.legal_moves:
        action = move.from_square * 64 + move.to_square
        legal_actions.append(action)
    legal_actions = np.array(legal_actions)
    if len(legal_actions) == 0:
        return None
    if np.random.rand() < epsilon:
        chosen_action = np.random.choice(legal_actions)
    else:
        q_values_legal = q_values[legal_actions]
        best_action = legal_actions[np.argmax(q_values_legal)]
        chosen_action = best_action
    from_sq, to_sq = action_to_move(chosen_action)
    return chess.Move(from_sq, to_sq)

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
    def __init__(self, num_actions=ACTION_SIZE):
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

def action_to_move(action):
    from_sq = action // 64
    to_sq = action % 64
    return from_sq, to_sq

def sample_action_from_qvalues(board, q_values, epsilon):
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return None
    legal_actions = []
    for move in legal_moves:
        action = move.from_square * 64 + move.to_square
        legal_actions.append(action)
    legal_actions = np.array(legal_actions)
    if np.random.rand() < epsilon:
        chosen_action = np.random.choice(legal_actions)
    else:
        legal_q_values = q_values[legal_actions]
        max_index = np.argmax(legal_q_values)
        chosen_action = legal_actions[max_index]
    return chosen_action

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

def update_model(dqn, dqn_target, optimizer, replay_buffer, batch_size=64, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states_t = torch.tensor(states, dtype=torch.float32).to(device)
    actions_t = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones_t = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = dqn(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = dqn_target(next_states_t).max(1)[0]

    target = rewards_t + gamma * next_q_values * (1 - dones_t)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def calculate_intermediate_reward(board, move, prev_fen=None, no_aggression_steps=None):
    # 中间奖励的计算逻辑，可根据需要调整
    attack = False
    reward = 0.0
    piece_values = {chess.PAWN: 0.15, chess.KNIGHT: 0.35, chess.BISHOP: 0.35, chess.ROOK: 0.55, chess.QUEEN: 1.1}

    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            reward += piece_values.get(captured_piece.piece_type, 0)
            attack = True

    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    if move.to_square in center_squares:
        reward += 0.25
        attack = True

    if board.gives_check(move):
        reward += 0.9
        attack = True

    current_fen = board.fen()
    if prev_fen is not None and current_fen == prev_fen:
        reward -= 0.6

    current_color = board.turn
    opponent_color = not current_color
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == opponent_color:
            if piece.piece_type == chess.QUEEN and board.is_attacked_by(current_color, square):
                reward += 0.35
                attack = True
            elif piece.piece_type == chess.ROOK and board.is_attacked_by(current_color, square):
                reward += 0.25
                attack = True
    if prev_fen is not None:
        board_before = chess.Board(prev_fen)
        # 计算我方在前一局面中的子总数和各类型数量
        own_pieces_before = {pt: len(board_before.pieces(pt, current_color)) for pt in chess.PIECE_TYPES}
        # 计算我方在当前局面中的子总数和各类型数量
        own_pieces_after = {pt: len(board.pieces(pt, current_color)) for pt in chess.PIECE_TYPES}

        # 识别被吃掉的子类型及数量
        lost_pieces = {}
        for pt in chess.PIECE_TYPES:
            lost = own_pieces_before.get(pt, 0) - own_pieces_after.get(pt, 0)
            if lost > 0:
                lost_pieces[pt] = lost

        # 根据被吃掉的子类型和数量扣分
        if lost_pieces:
            penalty = 0.0
            penalty_mapping = {
                chess.PAWN: 0.15,
                chess.KNIGHT: 0.35,
                chess.BISHOP: 0.35,
                chess.ROOK: 0.55,
                chess.QUEEN: 1.1
            }
            for pt, count in lost_pieces.items():
                penalty += penalty_mapping.get(pt, 0) * count
            reward -= penalty
            attack = False
    if not attack:
        reward -= 0.085
    if no_aggression_steps is not None and no_aggression_steps >= 10:
        reward -= 0.35

    return reward, attack

def calculate_intermediate_reward_opening(board, move, prev_board=None):
    # 示例奖励计算，可根据需要调整
    reward = 0.0
    moved_piece = board.piece_at(move.to_square)
    if moved_piece and moved_piece.piece_type == chess.PAWN:
        from_sq = move.from_square
        to_sq = move.to_square
        from_file = chess.square_file(from_sq)
        to_file = chess.square_file(to_sq)
        from_rank = chess.square_rank(from_sq)
        to_rank = chess.square_rank(to_sq)
        files = "abcdefgh"

        # 中央突破等策略
        if files[from_file] in ['d','e']:
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            if any(board.piece_at(sq) is None for sq in center_squares):
                reward += 0.2

        # 打开兵线
        file_pawns = [sq for sq in chess.SQUARES if chess.square_file(sq) == to_file and board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN]
        if len(file_pawns) == 1:
            reward += 0.2

        # 翼侧推进
        if files[from_file] in ['a','h']:
            reward += 0.1

        # 兵链对抗机会
        directions = [(-1,1),(1,1)] if board.turn == chess.WHITE else [(-1,-1),(1,-1)]
        for dx, dy in directions:
            check_file = to_file + dx
            check_rank = to_rank + dy
            if 0 <= check_file < 8 and 0 <= check_rank < 8:
                sq = chess.square(check_file, check_rank)
                p = board.piece_at(sq)
                if p and p.color != board.turn and p.piece_type == chess.PAWN:
                    reward += 0.2

        # 攻击对方王翼
        opponent_king_square = board.king(not board.turn)
        if opponent_king_square is not None:
            ok_file = chess.square_file(opponent_king_square)
            if ok_file in [5,6,7] and to_file in [5,6,7]:
                reward += 0.2

        # 中心转翼（简单化）
        if files[from_file] in ['d','e'] and files[to_file] in ['c','f']:
            reward += 0.15

        # 弃兵打开局面
        opponent_color = not board.turn
        attacked_by_opponent_pawn = False
        pawn_attack_directions = [(-1,1),(1,1)] if opponent_color else [(-1,-1),(1,-1)]
        for dx,dy in pawn_attack_directions:
            check_file = to_file + dx
            check_rank = to_rank + dy
            if 0 <= check_file < 8 and 0 <= check_rank < 8:
                sq = chess.square(check_file, check_rank)
                p = board.piece_at(sq)
                if p and p.color == opponent_color and p.piece_type == chess.PAWN:
                    attacked_by_opponent_pawn = True
                    break
        if attacked_by_opponent_pawn:
            reward += 0.15

    return reward

def play_game_and_collect_data(dqn, opponents, epsilon=0.1):
    """
    玩一局游戏，两个代理之间的对弈，并记录经验用于训练，同时动态引导行为。

    参数:
    - dqn: 主DQN模型
    - opponents: 对手列表，每个对手是一个元组 (name, model, predict_fn)
    - epsilon: 探索率

    返回:
    - transitions: 一局游戏中的所有转移 (state, action, reward, next_state, done)
    - game_result: 游戏结果 ("win", "loss", "draw")
    """
    # 确保每个对手元组至少包含三个元素 (name, model, predict_fn)
    assert all(len(opp) >= 3 for opp in opponents), "Each opponent tuple must have at least 3 elements (name, model, predict_fn)"
    
    board = chess.Board()
    done = False
    transitions = []
    game_result = None
    prev_fen = board.fen()
    no_aggression_steps = 0
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # 我方（DQN模型）行动
            state = board_state_to_rl_input(board)
            state_np = state[0].astype(np.float32)

            dqn.eval()
            with torch.no_grad():
                q_values = dqn(torch.tensor(state, dtype=torch.float32).to(device)).cpu().numpy().squeeze()

            action = sample_action_from_qvalues(board, q_values, epsilon)
            if action is None:
                break
            from_sq, to_sq = action_to_move(action)
            move = chess.Move(from_sq, to_sq)

            # 计算中间奖励
            intermediate_reward, attack = calculate_intermediate_reward(board, move, prev_fen, no_aggression_steps)
            if not attack:
                no_aggression_steps += 1
            else:
                no_aggression_steps = 0
                open_reward = calculate_intermediate_reward_opening(board, move)
                intermediate_reward += open_reward

            prev_fen = board.fen()
            board.push(move)

            if board.is_game_over():
                result = board.result()
                if result == "1-0":
                    final_reward = 8
                    game_result = "win"
                elif result == "0-1":
                    final_reward = -6.5
                    game_result = "loss"
                else:
                    final_reward = -4
                    game_result = "draw"
                done = True
                reward = intermediate_reward + final_reward
                next_state = np.zeros_like(state_np)
            else:
                reward = intermediate_reward
                next_state = board_state_to_rl_input(board)[0].astype(np.float32)

            transitions.append((state_np, action, reward, next_state, done))
            if done:
                break

        else:
            # 对手行动
            if len(opponents) == 2:
                # 使用两个对手之一
                if random.random() < 0.5:
                    opp_move = opponents[0][2](opponents[0][1], board)
                else:
                    opp_move = opponents[1][2](opponents[1][1], board)
            elif len(opponents) == 1:
                # 使用单一对手
                opp_move = opponents[0][2](opponents[0][1], board)
            else:
                opp_move = None

            if opp_move is None:
                break
            if isinstance(opp_move, chess.Move):
                opp_move = opp_move.uci()
            elif isinstance(opp_move, str):
                # 验证 UCI 格式
                try:
                    chess.Move.from_uci(opp_move)
                except:
                    logging.error(f"Invalid move from opponent: {opp_move}")
                    break
            else:
                logging.error(f"Unsupported move type from opponent: {opp_move}")
                break
            move = chess.Move.from_uci(opp_move)
            board.push(move)

            if board.is_game_over():
                result = board.result()
                if result == "1-0":
                    final_reward = 8
                    game_result = "win"
                elif result == "0-1":
                    final_reward = -6.5
                    game_result = "loss"
                else:
                    final_reward = -4
                    game_result = "draw"
                done = True
                if len(transitions) > 0:
                    s, a, r, ns, d = transitions[-1]
                    transitions[-1] = (s, a, r + final_reward, ns, True)
                break

    return transitions, game_result

def generate_opponent_sequence(total_episodes, opponents):
    """
    生成对手序列，根据各对手的比例

    参数:
    - total_episodes: 总训练轮数
    - opponents: 对手列表，每个对手是一个元组 (name, model, predict_fn, proportion)

    返回:
    - opponent_sequence: 对手序列列表，每个元素是 (name, model, predict_fn)
    """
    opponent_sequence = []
    for name, model, predict_fn, proportion in opponents:
        num_games = int(total_episodes * proportion)
        opponent_sequence.extend([(name, model, predict_fn)] * num_games)
    # 如果总局数不够，随机补充
    while len(opponent_sequence) < total_episodes:
        name, model, predict_fn, _ = random.choice(opponents)
        opponent_sequence.append((name, model, predict_fn))
    # 打乱顺序以增加随机性
    random.shuffle(opponent_sequence)
    return opponent_sequence

def main():
    dqn = DQN(num_actions=ACTION_SIZE).to(device)
    dqn_target = DQN(num_actions=ACTION_SIZE).to(device)

    # 从已训练好的模型继续训练（如果存在）
    final_model_path = "rl_dqn_chess_final.pth"
    if os.path.exists(final_model_path):
        print("Load from old model...")
        state_dict = torch.load(final_model_path, map_location=device)
        model_dict = dqn.state_dict()

        # 过滤掉不匹配的键，部分加载权重
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        dqn.load_state_dict(model_dict)

        # 同步目标网络
        dqn_target.load_state_dict(dqn.state_dict())
    else:
        print("No old model found. Training from scratch...")

    optimizer = optim.Adam(dqn.parameters(), lr=3e-2)
    
    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',            # 因为我们希望监控的指标是平均奖励，奖励越高越好
        factor=0.9,            # 当指标停止改善时，学习率将乘以 factor
        patience=1000,         # 如果指标在 patience 个步骤内没有改善，则调整学习率
        verbose=True,          # 打印学习率调整信息
        min_lr=1e-3             # 学习率的下限
    )
    
    replay_buffer = ReplayBuffer(capacity=4000000)

    episodes = 5000
    update_target_freq = 100
    epsilon = 0.4
    epsilon_min = 0.2
    epsilon_decay = 0.9985
    batch_size = 7000

    opponents = [
        ("self_play", dqn_target, predict_best_move_dqn, 0.1),       # 自对弈，占80%
        ("algorithm1", algorithm1, predict_best_move_model1, 0.5), # algorithm1，占10%
        ("algorithm2", algorithm2, predict_best_move_model2, 0.4)  # algorithm2，占10%
    ]

    # 生成对手序列
    opponent_sequence = generate_opponent_sequence(episodes, opponents)

    # 初始化统计变量
    win_count = 0
    loss_count = 0
    draw_count = 0
    episode_rewards = []
    win_rates = []
    avg_rewards = []
    opponent_counts = {"self_play": 0, "algorithm1": 0, "algorithm2": 0}
    learning_rates = []

    print("Start Training...")
    with tqdm(range(episodes), desc="Training Episodes") as pbar:
        for ep in pbar:
            current_opponent = opponent_sequence[ep]
            opponent_name, opponent_model, opponent_predict_fn = current_opponent

            # 记录对手次数
            opponent_counts[opponent_name] += 1

            # 传递整个 current_opponent 元组
            transitions, game_result = play_game_and_collect_data(dqn, [current_opponent], epsilon=epsilon)
            total_reward = sum([t[2] for t in transitions])
            episode_rewards.append(total_reward)

            # 根据结果统计
            if game_result == "win":
                win_count += 1
            elif game_result == "loss":
                loss_count += 1
            elif game_result == "draw":
                draw_count += 1

            for t in transitions:
                replay_buffer.push(*t)

            # 多次更新模型
            for _ in range(10):
                update_model(dqn, dqn_target, optimizer, replay_buffer, batch_size=batch_size, gamma=0.99)

            # 同步目标网络
            if (ep + 1) % update_target_freq == 0:
                dqn_target.load_state_dict(dqn.state_dict())

            # 计算Win Rate和平均Reward
            total_games = win_count + loss_count + draw_count
            current_win_rate = win_count / total_games if total_games > 0 else 0.0
            avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)

            win_rates.append(current_win_rate)
            avg_rewards.append(avg_reward_100)

            # 步进调度器，传入监控指标（avg_reward_100）
            scheduler.step(avg_reward_100)

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 更新epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # 更新进度条的后缀信息
            pbar.set_postfix({
                "Win Rate": f"{current_win_rate:.2f}",
                "Avg Reward(100)": f"{avg_reward_100:.2f}",
                "Epsilon": f"{epsilon:.3f}",
                "LR": f"{current_lr:.6f}",
                "Buffer": len(replay_buffer),
                "Self_Play": opponent_counts["self_play"],
                "Algo1": opponent_counts["algorithm1"],
                "Algo2": opponent_counts["algorithm2"]
            })

    # 训练结束后保存模型
    continued_model_path = "rl_dqn_chess_final.pth"
    torch.save(dqn.state_dict(), continued_model_path)

    # 绘制Win Rate、平均Reward和学习率曲线
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(win_rates, label='Win Rate')
    plt.title("Win Rate over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(avg_rewards, label='Average Reward (last 100 games)', color='orange')
    plt.title("Average Reward (last 100 games) over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', color='green')
    plt.title("Learning Rate over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 输出对弈比例
    print("Opponent Counts:")
    total_opponents = sum(opponent_counts.values())
    for name, count in opponent_counts.items():
        proportion = count / total_opponents
        print(f"{name}: {count} games, Proportion: {proportion:.2%}")

    print(f"Training finished and model saved as {continued_model_path}.")

if __name__ == "__main__":
    main()


