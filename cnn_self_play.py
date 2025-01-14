import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Custom neural network model
class ChessCNN(nn.Module):
    def __init__(self, num_classes=80):
        super(ChessCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),  # Global pooling to 1x1
            nn.Flatten(),  # Flatten to a one-dimensional vector
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Adjust dimensions to [batch_size, channels, height, width]
        return self.model(x)

# Load model
def load_pytorch_model(file_path, num_classes):
    model = ChessCNN(num_classes)  # Create model instance

    # Load model weights with partial matching allowed
    checkpoint = torch.load(file_path, map_location=torch.device('cuda'))
    
    # Load with strict=False to allow partial matching of weights
    model.load_state_dict(checkpoint, strict=False)
    
    model.eval()  # Set to evaluation mode
    return model

# Initialize board state
def initial_board_state():
    board = chess.Board()
    return board

# Convert board state to neural network input format (8x8x12)
def board_state_to_input(board):
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

# Get legal moves
def get_legal_moves(board):
    legal_moves = [move.uci() for move in board.legal_moves]
    return legal_moves

# Predict the best move using the model
def predict_best_move(model, board_state):
    """
    Predict the move and select the best move
    """
    legal_moves = get_legal_moves(board_state)

    if not legal_moves:
        return None  # Return None if there are no legal moves

    best_move = None
    best_move_value = -float('inf')  # Store the value of the best move

    # Accumulate legal moves and compute the predicted value for each move
    for move in legal_moves:
        # Convert board state to model input
        board_input = board_state_to_input(board_state)
        board_input = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0)  # Convert to PyTorch tensor

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

# Apply move
def apply_move(board, move):
    chess_move = chess.Move.from_uci(move)  # Convert the move string to a Move object on the board
    if chess_move in board.legal_moves:
        board.push(chess_move)  # Execute the move
    else:
        print("Invalid move:", move)
    return board

# Evaluate board score (scores for white and black)
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

# Log game state and scores
def log_game_state(board, white_score, black_score, move_number, file_path="game_log0.txt"):
    with open(file_path, "a") as file:
        file.write(f"Move {move_number}:\n")
        file.write(str(board) + "\n")
        file.write(f"White score: {white_score}\n")
        file.write(f"Black score: {black_score}\n")
        file.write("-" * 4 + "\n")

# Main function: Play against the model
def play_game_M2M(model):
    board_state = initial_board_state()
    move_number = 1

    while not board_state.is_game_over():  # Check if the game is over
        white_score, black_score = evaluate_board(board_state)  # Calculate scores
        print_board_state(board_state, white_score, black_score, move_number)

        # Select the best move
        best_move = predict_best_move(model, board_state)
        if best_move:
            print(f"Best move: {best_move}")
            board_state = apply_move(board_state, best_move)
        else:
            print("No valid moves available.")
            break

        # Log the current game state and scores
        log_game_state(board_state, white_score, black_score, move_number)

        move_number += 1

    print("Game over!")
    print(game_over_reason(board_state), '\n')
    print_winner(board_state)

# Print the board state
def print_board_state(board, white_score, black_score, move_number):
    print(f"\nMove {move_number}:")
    print(board)
    print(f"White score: {white_score}")
    print(f"Black score: {black_score}")
    print("---" * 6)
    time.sleep(1)  # Add a 1-second delay

# Determine the reason for game over
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

# Print the winner
def print_winner(board_state):
    result = board_state.result()  # Get the game result
    if result == '1-0':
        print("White wins!")
    elif result == '0-1':
        print("Black wins!")
    elif result == '1/2-1/2':
        print("It's a draw!")

# Load the model
model = load_pytorch_model("chess_cnn_model_ql.pth", num_classes=80)  # Replace with your model path

# Start the game against the model
play_game_M2M(model)
