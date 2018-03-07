import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import chess
import chess.svg

board = chess.Board()

board.push_san("b4")
board.push_san("Na6")
print(board)



file = open("current_board.html", "w")

printthis = '''
<html>
<head><meta http-equiv="refresh" content="5"></head>

<body>''' + chess.svg.board(board) + ''''</body>

</html>'''



file.write(printthis)



file.close()