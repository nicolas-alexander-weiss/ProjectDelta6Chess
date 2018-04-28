
import time
import os.path

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib



training_data_beg = np.array([
                     # board
                     4., 2., 3., 5., 6., 3., 2., 4., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -1., -1., -1., -1., -1.,
                     -1., -1., -4., -2., -3., -5., -6., -3., -2., -4.,
                     # flags
                     1, 1, 1, 1, 0,
                     # turn indicator
                     1
                     ])


nn_params = {"hdl_sizes": (1000,1000,1000),
             "activation": "relu",
             "solver": "sgd",
             "alpha": 0.00001,  # regularization parameter!
             "learning_rate": "constant",
             "max_iter": 1,
             "rand_seed": int(time.time()),
             "warm_start": True,
             "verbose": False}

stats_index = {"num_trained_examples": 0}


class NN_Regressor:
    def __init__(self, clf_name, save=False):
        self.clf_name = clf_name
        self.clf_file_name = "src/chess/recent/clfs/" + clf_name + ".pkl"
        self.stats_file_name = "src/chess/recent/clfs/" + clf_name + ".stats.pkl"
        self.save = save

        self.clf = self.load_clf()
        if self.clf is None:
            self.clf = MLPRegressor(hidden_layer_sizes=nn_params["hdl_sizes"], activation=nn_params["activation"],
                                    solver = nn_params["solver"], alpha=nn_params["alpha"],
                                    learning_rate=nn_params["learning_rate"], max_iter=nn_params["max_iter"],
                                    random_state=nn_params["rand_seed"], warm_start=nn_params["warm_start"],
                                    verbose=nn_params["verbose"])
            self.clf.fit([training_data_beg], [0])
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

    def save_clf_and_stats(self):
        if nn_params["verbose"]:
            print("saving clf")
        joblib.dump(self.clf, filename=self.clf_file_name)
        if nn_params["verbose"]:
            print("saving stats")
        joblib.dump(self.stats, filename=self.stats_file_name)

    def train_clf(self, X, y):
        #training

        if nn_params["verbose"]:
            # self.clf.set_params(verbose = True)
            print("begin training")
            beg = time.time()

        self.clf.fit(X,y)

        if nn_params["verbose"]:
            end = time.time()
            print("done training,", end-beg, "sec")

        # saving
        self.stats[stats_index["num_trained_examples"]] += len(X)
        if self.save:
            self.save_clf_and_stats()
