import chess
import numpy as np

sym_to_int = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN, "K": chess.KING,
              "p": - chess.PAWN, "n": - chess.KNIGHT, "b": - chess.BISHOP, "r": - chess.ROOK, "q": - chess.QUEEN,
              "k": - chess.KING}


def board_to_vector(board):
    board_vect = np.zeros(64)

    for i in range(0,64,1):
        piece = board.piece_at(i)
        if piece is None:
            continue
        board_vect[i] = sym_to_int[piece.symbol()]

    return board_vect


class ChessAI:

    def __init__(self):
        self.load_tensor_flow_model()

    def load_tensor_flow_model(self):
        pass

    def train_tensor_flow_model(self, game_data):
        X, y = self.map_data(game_data)

    def map_data(self, game_data):
        pass

    # game methods:
    def evaluate_board(self, board):
        pass
