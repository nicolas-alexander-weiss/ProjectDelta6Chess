import numpy as np

EMPTY = 0

INVERSE = -1

BLACK = -1
WHITE = 1

PAWN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
QUEEN = 5
KING = 6


class Board:
    def __init__(self):
        self.__board = np.array([[2, 3, 4, 5, 6, 4, 3, 2],
                                 [1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [-1, -1, -1, -1, -1, -1, -1, -1],
                                 [-2, -3, -4, -5, -6, -4, -3, -2]])
        self.__movedRooks = np.array([0, 0, 0, 0])  # wRook1,wRook2, bRook1, bRook2
        self.__movedKings = np.array([0, 0])

    def print(self):
        for i in range(0, 8, 1):
            for j in range(0, 8, 1):
                print("%3d" % (self.__board[7 - i][j]), end='')

            print()
        print()

    def update(self, a1, a2, b1, b2):
        if a1 < 0 or a1 > 7 or a2 < 0 or a2 > 7 or b1 < 0 or b1 > 7 or b2 < 0 or b2 > 7:
            return False

        self.__board[b1][b2] = self.__board[a1][a2]
        self.__board[a1][a2] = 0

        return True

    # returns array of [[a1,a2,b1,b2],[a1,....],[...],]
    def get_possible_moves(self, color):
        pm = []
        if color != BLACK and color != WHITE:
            raise (ValueError, "Color specified as " + color)
        for i in range(0,8,1):
            for j in range(0,8,1):
                piece = self.__board[i][j]
                if color == BLACK and piece >= BLACK * KING and piece <= BLACK * PAWN \
                        or color == WHITE and piece >= WHITE * PAWN and piece <= BLACK * PAWN :
                    if piece == color * PAWN:
                        if(self.__board[i + color][j] == EMPTY):
                            pm.append(np.array([i,j,i + color,j]))



class Game:
    def __init__(self):
        self.board = Board()
        self.round = 0
        self.atTurn = BLACK


board = Board()
board.print()

print(board.get_possible_moves(BLACK))