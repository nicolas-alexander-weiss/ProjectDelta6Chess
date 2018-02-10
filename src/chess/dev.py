import numpy as np

class Board:
    __board = np.array([[2,3,4,5,6,4,3,2],[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,-1,-1,-1],[-2,-3,-4,-5,-6,-4,-3,-2]])

    def printBoard(self):
        for i in range(0,8,1):
            for j in range(0,8,1):
                print("%3d" % (self.__board[7 - i][j]), end='')

            print("\n")


board = Board()
board.printBoard()
