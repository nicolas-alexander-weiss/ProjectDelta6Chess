import numpy as np

class Board:

    __board = np.array([[2,3,4,5,6,4,3,2],[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[-1,-1,-1,-1,-1,-1,-1,-1],[-2,-3,-4,-5,-6,-4,-3,-2]])

    def print(self):
        for i in range(0,8,1):
            for j in range(0,8,1):
                print("%3d" % (self.__board[7 - i][j]), end='')

            print()
        print()

    def update(self, a1, a2, b1, b2):
        if a1 < 0 or a1 > 7 or a2 < 0 or a2 > 7 or b1 < 0 or b1 > 7 or b2 < 0 or b2 > 7:
            return False

        self.__board[b1][b2] = self.__board[a1][a2]
        self.__board[a1][a2] = 0

        return True



board = Board()
board.update()
board.update(0, 0, 7, 7)
board.update()

val = board.update(10, 0, 7, 7)
board.update()
print (val)