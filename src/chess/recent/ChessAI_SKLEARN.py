import time
import os.path

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import chess

import numpy as np

import src.chess.recent.ChessUtils as ChessUtils

nn_params = {"hdl_sizes": (100, 100, 100),
             "activation": "logistic",
             "solver": "adam",
             "alpha": 0.00001,  # regularization parameter!
             "learning_rate": "constant",
             "rand_seed": int(time.time()),
             "warm_start": True,
             "verbose": True}

stats_index = {"num_trained_examples": 0}

class ChessAI:
    def __init__(self, clf_name, save=False):
        self.clf_name = clf_name
        self.clf_file_name = clf_name + ".pkl"
        self.stats_file_name = clf_name + ".stats.pkl"

        self.save = save

        self.clf = self.load_clf()
        if self.clf is None:
            self.clf = MLPClassifier(solver=nn_params["solver"], alpha=nn_params["alpha"],
                                     hidden_layer_sizes=nn_params["hdl_sizes"], random_state=nn_params["rand_seed"],
                                     warm_start=nn_params["warm_start"], verbose=nn_params["verbose"])
        self.stats = self.load_stats()
        if self.stats is None:
            self.stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def load_clf(self):
        # load classifier
        if os.path.isfile(self.clf_file_name):
            return joblib.load(self.clf_file_name)
        else:
            return None

    def load_stats(self):
        # load stats
        if os.path.isfile(self.stats_file_name):
            return joblib.load(self.stats_file_name)
        else:
            return None

    def train_clf(self, X, y, manual = False, mini_batch_size = 10, max_itr = 1):
        temp_iter = self.clf.get_params()["max_iter"]
        if manual:
            self.clf.set_params(max_iter=max_itr)

        print("begin training")
        beg = time.time()
        if not manual:
            self.clf.fit(X, y)
        else:
            num_batches = len(X) //  mini_batch_size
            x_batches = self.make_batches(X, mini_batch_size)
            y_batches = self.make_batches(y, mini_batch_size)
            for i in range(num_batches):
                print("ex",i, "of", num_batches)
                self.clf.fit(x_batches[i], y_batches[i])#.reshape((1,-1)),y_batches[i].reshape((1,-1)))
        end = time.time()
        print("done training,", end-beg, "sec")

        self.clf.set_params(max_iter=temp_iter)

        self.stats[stats_index["num_trained_examples"]] += len(X)
        if self.save:
            self.save_clf_and_stats()

    def make_batches(self, array, mini_batch_size):
        top = -1
        last_ind = len(array) - 1
        batches = []

        while top < last_ind:
            diff = last_ind - top
            if diff <= mini_batch_size:
                batches += [array[top + 1: top + diff + 1]]
                top += diff
            else:
                batches += [array[top + 1: top + mini_batch_size + 1]]
                top += mini_batch_size
        return batches

    def save_clf_and_stats(self):
        print("saving clf")
        joblib.dump(self.clf, filename=self.clf_file_name)
        print("saving stats")
        joblib.dump(self.stats, filename=self.stats_file_name)

    def predict_win_prob_white(self, board):
        return self.clf.predict_proba(board)


if __name__ == "__main__":
    ai = ChessAI("test_clf", True)

    #boards_resign = np.load("boards_resign.npy")
    #boards_mate = np.load("boards_mate.npy")
    #print(ai.clf.get_params())
    # ai.clf.set_params(max_iter=1)

    #ai.train_clf(boards_mate[:,1:],boards_mate[:,0], manual=True, mini_batch_size=10)

    #print(ai.make_batches([1,2,3,4,5], 1))

    moves = "e4 e5 Nf3 Nc6 Bc4 Nf6 Nc3 Bc5 O-O O-O Ne1 Ne8 Qf3 Qg5 d4 Qg6 dxc5 Nd4 Qd1 d6 g3 Bh3 Nf3 Bxf1 Qxf1 Nxf3+ Kg2 Nh4+ Kh1"
    moves += " Nf3 Nd5 Qe6 Nxc7 Qc8 Nxa8 Qxa8 cxd6 Nxd6 Be2 Nd4 Bd3 Qc8 c3 Ne6 Qe1 Nc4 Bxc4 Qxc4 b3 Qc6 Ba3 Rd8 Be7 Rd7 Ba3 Ng5 Qd2"
    moves += " Rxd2 Kg2 Qxe4+ Kf1 Qe2+ Kg2 Qxf2+ Kh1 Qxh2#"
    moves = moves.split(" ")

    board = chess.Board()

    for move in moves:
        board.push_san(move)
    print(board)
    #board.push_san("e4")
    #board.push_san("e6")
    #board.push_san("Nh3")
    board_vect = ChessUtils.board_to_vector(board).reshape((1,-1))
    print(board_vect)
    print(ai.predict_win_prob_white(board_vect))

# looks like pred[1] seems to be whites winning prob
# pred[0] black? YES!


