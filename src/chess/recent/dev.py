import numpy as np

import time

import chess
import chess.svg

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def print_board_to_file():
    board = chess.Board()

    board.push_san("b4")
    board.push_san("Na6")
    board.push_san("e4")
    print(board)

    file = open("current_board.html", "w")

    print_this = "<html><head><meta http-equiv='refresh' content='5'></head><body>"\
                 + chess.svg.board(board) + "</body></html>"

    file.write(print_this)

    file.close()

def try_sklearn():

    boards_mate = np.load("boards_mate.npy")


    X = boards_mate[0:100,1:]
    y = boards_mate[0:100,0]

    print("size ", len(boards_mate))

    print("Start Training")
    clf = MLPClassifier(solver="adam", alpha=0.0001, hidden_layer_sizes=(100,100,100), random_state=1)

    start = time.time()
    clf.fit(X,y)
    end = time.time()
    print("Training Done,", end-start, "seconds.")

    # print(clf.get_params())
    # print("coefs_\n",clf.coefs_)
    # print("intercepts_\n", clf.intercepts_)


print_board_to_file()