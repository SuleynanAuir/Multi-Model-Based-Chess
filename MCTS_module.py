from math import log, sqrt
from random import random
import chess
import numpy as np
import torch
import torch.nn as nn
from model_training_mcts_batch import ChessModel


class Node:
    def __init__(self, state, move=None, parent=None):
        self.move = move
        self.state = state
        self.parent = parent
        self.unexplored_moves = list(state.legal_moves)  # 未探索的合法棋步
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, state, move):
        child_node = Node(state, move, self)
        self.children.append(child_node)
        self.unexplored_moves.remove(move)
        return child_node

    def UCT_select_child(self, model):
        best_value = float('-inf')
        best_child = None
        for child in self.children:
            win_rate = child.wins / (child.visits + 1e-6)  # 避免除零
            exploration = sqrt(2 * log(self.visits + 1) / (child.visits + 1e-6))
            heuristic = self.heuristic(child.state, model)
            value = win_rate + exploration + heuristic

            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def split_dims(board):
        board3d = np.zeros((12, 8, 8), dtype=np.int8)
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                row, col = divmod(square, 8)
                board3d[piece - 1, row, col] = 1
            for square in board.pieces(piece, chess.BLACK):
                row, col = divmod(square, 8)
                board3d[5 + piece, row, col] = 1
        return board3d
    
    def heuristic(self, state, model):
        state_tensor = torch.tensor(Node.split_dims(state), dtype=torch.float32).unsqueeze(0).to(torch.device)
        with torch.no_grad():
            q_values = model(state_tensor)  # CNN模型输出棋步概率
        return torch.max(q_values).item()

    def update(self, result):
        self.visits += 1
        self.wins += result
    


class MCTS_CNN_Model(nn.Module):
    def __init__(self, conv_size, conv_depth, num_classes):
        super(MCTS_CNN_Model, self).__init__()
        self.cnn_model = ChessModel(conv_size=conv_size, conv_depth=conv_depth, num_classes=num_classes)
        self.rootnode = None

    def set_root_node(self, rootstate):
        self.rootnode = Node(state=rootstate)

    def forward(self, x):
        return self.cnn_model(x)

    def select_move_based_on_policy(self, policy_probs):
        legal_moves = list(self.rootnode.state.legal_moves)  # 获取合法棋步
        legal_move_probs = [policy_probs[i] for i in range(len(legal_moves))]  # 提取合法棋步的概率
        selected_index = random.choices(range(len(legal_moves)), weights=legal_move_probs, k=1)[0]  # 按概率选择
        return selected_index


    def UCT_search(self, itermax, depthmax):
        rootnode = self.rootnode
        for i in range(itermax):
            node = rootnode
            state = rootnode.state.copy()

            while node.unexplored_moves == [] and node.children != []:
                node = node.UCT_select_child(self.cnn_model)
                state.push(node.move)

            if node.unexplored_moves != []:
                m = random.choice(node.unexplored_moves)
                state.push(m)
                node = node.add_child(state, m)

            while state.legal_moves != []:
                move_probs = torch.softmax(self.cnn_model(Node.board_state_to_input(state)), dim=-1).squeeze()
                move_index = self.select_move_based_on_policy(move_probs)
                legal_moves = list(state.legal_moves)
                state.push(legal_moves[move_index])

            result = 1 if self.cnn_model(Node.board_state_to_input(state)).item() > 0 else 0
            node.update(result)

        return self.select_move_based_on_policy(
            torch.softmax(self.cnn_model(Node.board_state_to_input(self.rootnode.state)), dim=-1)
        )
    # 加载模型参数并处理缺失层
# 加载模型参数并处理缺失层
def load_model_with_missing_layers(model, checkpoint):
    try:
        # 使用 strict=False 以允许加载时丢失的层
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    # 处理缺失的层
    missing_layers = set(model.state_dict().keys()) - set(checkpoint['model_state_dict'].keys())
    if missing_layers:
        print("Warning: Missing layers detected. Initializing missing layers.")
        for layer in missing_layers:
            if 'conv_layers' in layer:
                print(f"Reinitializing missing layer: {layer}")
                # 重新初始化缺失的卷积层
                model.cnn_model.conv_layers.apply(init_weights)
            else:
                print(f"Layer {layer} is missing and will be initialized.")
                # 重新初始化其他缺失的层
                model.apply(init_weights)

def init_weights(layer):
    """初始化卷积层的权重"""
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)









