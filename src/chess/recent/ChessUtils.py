import chess
import numpy as np

sym_to_int = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN,
                  "K": chess.KING,
                  "p": - chess.PAWN, "n": - chess.KNIGHT, "b": - chess.BISHOP, "r": - chess.ROOK, "q": - chess.QUEEN,
                  "k": - chess.KING}


def board_to_vector(board):
    board_vect = np.zeros(64)

    for i in range(0, 64, 1):
        piece = board.piece_at(i)
        if piece is None:
            continue
        board_vect[i] = sym_to_int[piece.symbol()]

    return board_vect

def vector_to_board(vector):
    if len(vector) != 64:
        return None
    board_string = ""
    for i in range(0,64,1):
        if vector[i] == 0:
            board_string += ""
    board_string +=  "w KQkq - 0 1"

if __name__ == "__main__":
    board = chess.Board()
    print(board_to_vector(board))