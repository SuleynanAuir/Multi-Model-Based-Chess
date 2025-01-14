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


from battle import (
    board_state_to_input_model1,
    board_state_to_input_model2,
    load_algorithm1_model,
    load_algorithm2_model,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载对手模型
algorithm1, device1 = load_algorithm1_model("mcts_cnn_model.pth", num_classes=64)
algorithm2 = load_algorithm2_model("chess_cnn_model.pth", num_classes=80)

ACTION_SIZE = 64 * 64

def predict_best_move_model1(model, board_state):
    legal_moves = list(get_legal_moves(board_state))
    if not legal_moves:
        return None
    board_input = board_state_to_input_model1(board_state)
    board_input = torch.tensor(board_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(board_input)
        move_probs = torch.softmax(predictions, dim=-1).squeeze().cpu().numpy()
    if len(move_probs) < len(legal_moves):
        move_probs = move_probs[:len(legal_moves)]
    move_dict = {m: p for m, p in zip(legal_moves, move_probs)}
    best_move = max(move_dict, key=move_dict.get)
    return best_move

def predict_best_move_model2(model, board_state):
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
    move_values = {m: v for m, v in zip(legal_moves, prediction)}
    best_move = max(move_values, key=move_values.get)
    return best_move

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

class DQN(nn.Module):
    def __init__(self, num_actions=ACTION_SIZE):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

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

def get_legal_moves(board):
    legal_moves = [move.uci() for move in board.legal_moves]
    return legal_moves

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

def calculate_intermediate_reward(board, move,prev_fen=None,no_aggression_steps = None):
    # 加强进攻性奖励：
    # 棋子价值（加大比例）：pawn=5, knight/bishop=15, rook=25, queen=45
    # 将军奖励=+10
    # 没有进攻动作(不吃子不将军)则-2
    attack = False
    reward = 0.0
    piece_values = {chess.PAWN: 10, chess.KNIGHT: 25, chess.BISHOP: 25, chess.ROOK: 30, chess.QUEEN: 50}

    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            reward += piece_values.get(captured_piece.piece_type, 0)
            attack =True
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    if move.to_square in center_squares:
        reward += 2
        attack = True
    if board.gives_check(move):
        reward += 20
        attack = True
    current_fen = board.fen()
    if prev_fen is not None and current_fen == prev_fen:
        # 如果当前局面和上一回合局面相同，惩罚
        reward -= 5.0
    
    current_color = board.turn
    opponent_color = not current_color
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == opponent_color:
            # 若对方有后或车在这个格子且被当前方攻击
            if piece.piece_type == chess.QUEEN and board.is_attacked_by(current_color, square):
                reward += 5  # 威胁对方后额外奖励
                attack = True
            elif piece.piece_type == chess.ROOK and board.is_attacked_by(current_color, square):
                reward += 3  # 威胁对方车额外奖励
                attack = True
    if attack == False:
        reward -= 1.3
    if no_aggression_steps is not None and no_aggression_steps >= 10:
        reward -= 10.0
    
    return reward ,attack
def calculate_intermediate_reward_opening(board, move, prev_board=None):
    """
    根据打开局面的策略给予额外奖励的中间奖励函数示例。
    prev_board: 走子前的棋盘状态(Chess.Board的副本)，用于比较变化。
    注：此函数假定您已在基础奖励（如吃子、将军、威胁子力）后调用，可以在原有基础奖励计算完后再调用此函数叠加奖励。
    """

    reward = 0.0

    # 基本参数
    piece = board.piece_at(move.to_square)
    moved_piece = board.piece_at(move.to_square)
    if moved_piece and moved_piece.piece_type == chess.PAWN:
        from_sq = move.from_square
        to_sq = move.to_square
        from_file = chess.square_file(from_sq)   # 0 to 7, 对应a-h
        to_file = chess.square_file(to_sq)
        from_rank = chess.square_rank(from_sq)   # 0 to 7, 对应1-8
        to_rank = chess.square_rank(to_sq)
        files = "abcdefgh"

        # 1. 中央突破：如果走的是中央兵(d或e列)，并完成推进或吃子
        if files[from_file] in ['d','e']: 
            # 条件示意：如该走子使得中心更开放，可加分
            # 简化逻辑：如果是前进一格或两格，且此后中心格子如 d4,e4,d5,e5不再有己方或对方兵阻挡，可给予奖励
            # 实际可进一步检查中心格子的兵情况
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            if any(board.piece_at(sq) is None for sq in center_squares):
                reward += 2.0  # 中央突破小奖励

        # 2. 打开兵线：检查该兵走子后是否产生了某条垂直线无兵（如之前有阻挡，现在没有）
        # 简化：若该兵吃子或推进导致该列上双方无兵，则加分
        file_pawns = [sq for sq in chess.SQUARES if chess.square_file(sq) == to_file and board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN]
        if len(file_pawns) == 1:  # 仅当前这个兵在此列或无兵说明线更开放了
            reward += 1.5

        # 3. 翼侧推进：如果是a列(0)或h列(7)的兵向前推进，加微小奖励，鼓励从侧翼打开局面
        if files[from_file] in ['a','h']:
            reward += 1.0

        # 4. 突破阻挡兵链：判断是否打破了对方的兵链
        # 兵链判定较复杂，这里仅示意：如果该兵推进后直接与对方兵对峙（对方兵在相邻对角格）
        # 且此举可能为打开新通路也可给予奖励
        # 简化逻辑：如果推进后相邻对角格存在对方兵，可视为兵链对抗机会增大
        directions = [(-1,1),(1,1)] if board.turn == chess.WHITE else [(-1,-1),(1,-1)]
        for dx,dy in directions:
            check_file = to_file+dx
            check_rank = to_rank+dy
            if 0 <= check_file < 8 and 0 <= check_rank < 8:
                sq = chess.square(check_file, check_rank)
                p = board.piece_at(sq)
                if p and p.color != board.turn and p.piece_type == chess.PAWN:
                    reward += 2.0  # 对兵链施压奖励

        # 5. 攻击国王翼：判断对方国王位置，对方国王通常在g、h文件或已易位后位于国王翼
        # 简单做法：找到对方国王位置，若此兵走动靠近或打开其侧翼线加奖励
        opponent_king_square = board.king(not board.turn)
        if opponent_king_square is not None:
            ok_file = chess.square_file(opponent_king_square)
            # 如果对方王在g,h侧(如白方王在g1,h1或黑方王在g8,h8), 且我们在f,g,h列推进
            if ok_file in [5,6,7] and to_file in [5,6,7]:
                # 在国王翼推进兵
                reward += 2.0

        # 6. 中心转翼：如果之前中心已控制，如己方在中心有兵控制，现在将战火转向侧翼（如从e4转推c5）
        # 简化逻辑：如果移动的兵不是原先的中央线兵，而是从中心区域(如d,e文件)向c,f文件推进，可视为转移战斗重心
        # 实际实现需更详细判断，这里仅作演示
        if files[from_file] in ['d','e'] and files[to_file] in ['c','f']:
            reward += 1.5

        # 7. 弃兵突破：如果推进后该兵可以被对方兵轻易吃掉(下一步对方可直接吃不亏)，则可能是弃兵突破
        # 简化逻辑：如果推进的位置能被对方兵直接吃（对方有兵能攻击该格子），给少量奖励作为战略性牺牲的鼓励
        opponent_color = not board.turn
        attacked_by_opponent_pawn = False
        pawn_attack_directions = [(-1,1),(1,1)] if opponent_color else [(-1,-1),(1,-1)]
        for dx,dy in pawn_attack_directions:
            check_file = to_file+dx
            check_rank = to_rank+dy
            if 0 <= check_file < 8 and 0 <= check_rank < 8:
                sq = chess.square(check_file, check_rank)
                p = board.piece_at(sq)
                if p and p.color == opponent_color and p.piece_type == chess.PAWN:
                    attacked_by_opponent_pawn = True
                    break

        if attacked_by_opponent_pawn:
            reward += 1.5  # 弃兵以打开局面的小额奖励

    return reward

def play_game_and_collect_data(dqn, opponent_model, epsilon=0.1):
    board = chess.Board()
    done = False
    transitions = []
    game_result = None  # win/loss/draw
    prev_fen = board.fen()
    no_aggression_steps = 0
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            state = board_state_to_rl_input(board)
            state_np = state[0].astype(np.float32)

            dqn.eval()
            with torch.no_grad():
                q_values = dqn(torch.tensor(state, dtype=torch.float32).to(device)).cpu().numpy().squeeze()

            action = sample_action_from_qvalues(board, q_values, epsilon)
            if action is None:
                # 无合法动作
                break
            from_sq, to_sq = action_to_move(action)
            move = chess.Move(from_sq, to_sq)

            intermediate_reward ,attack = calculate_intermediate_reward(board, move, prev_fen)
            if attack == False:
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
                    final_reward = 300
                    game_result = "win"
                elif result == "0-1":
                    final_reward = -150
                    game_result = "loss"
                else:
                    final_reward = -70
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
            if random.random() < 0.5:
                opp_move = predict_best_move_model1(opponent_model[0], board)
            else:
                opp_move = predict_best_move_model2(opponent_model[1], board)
            if opp_move is None:
                break
            move = chess.Move.from_uci(opp_move)
            board.push(move)

            if board.is_game_over():
                result = board.result()
                if result == "1-0":
                    final_reward = 300
                    game_result = "win"
                elif result == "0-1":
                    final_reward = -150
                    game_result = "loss"
                else:
                    final_reward = -70
                    game_result = "draw"
                done = True
                if len(transitions) > 0:
                    s, a, r, ns, d = transitions[-1]
                    transitions[-1] = (s, a, r + final_reward, ns, True)
                break

    return transitions, game_result

def main():
    dqn = DQN(num_actions=ACTION_SIZE).to(device)
    dqn_target = DQN(num_actions=ACTION_SIZE).to(device)

    # 从已训练好的模型继续训练（如果存在）
    final_model_path = "rl_dqn_chess_final.pth"
    if os.path.exists(final_model_path):
        print("load from old model")
        dqn.load_state_dict(torch.load(final_model_path, map_location=device))
        dqn_target.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=4e-4)
    replay_buffer = ReplayBuffer(capacity=100000)

    episodes = 150000
    update_target_freq = 100
    epsilon = 3.0
    epsilon_min = 0.1
    epsilon_decay = 0.99995
    batch_size = 64

    opponents = (algorithm1, algorithm2)

    win_count = 0
    loss_count = 0
    draw_count = 0
    episode_rewards = []
    win_rates = []
    avg_rewards = []

    # 使用tqdm进度条来展示训练过程
    with tqdm(range(episodes), desc="Training Episodes") as pbar:
        for ep in pbar:
            transitions, game_result = play_game_and_collect_data(dqn, opponents, epsilon=epsilon)
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

            for _ in range(10):
                update_model(dqn, dqn_target, optimizer, replay_buffer, batch_size=batch_size, gamma=0.99)

            if (ep + 1) % update_target_freq == 0:
                dqn_target.load_state_dict(dqn.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            total_games = win_count + loss_count + draw_count
            current_win_rate = win_count / total_games if total_games > 0 else 0.0
            avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)

            win_rates.append(current_win_rate)
            avg_rewards.append(avg_reward_100)
            # 使用tqdm的动态显示功能更新进度描述
            pbar.set_postfix({
                "Win Rate": f"{current_win_rate:.2f}",
                "Avg Reward(100)": f"{avg_reward_100:.2f}",
                "Epsilon": f"{epsilon:.3f}",
                "Buffer": len(replay_buffer)
            })

    # 训练结束后保存模型
    continued_model_path = "rl_dqn_chess_final.pth"
    torch.save(dqn.state_dict(), continued_model_path)

    # 绘制Win Rate和平均Reward的变化趋势图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(win_rates, label='Win Rate')
    plt.title("Win Rate over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards, label='Average Reward (last 100 games)', color='orange')
    plt.title("Average Reward (last 100 games) over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Training finished and model saved as {continued_model_path}.")

if __name__ == "__main__":
    main()
