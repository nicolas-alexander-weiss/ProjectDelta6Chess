import chess
import numpy as np
import tensorflow as tf

class ChessAI:

    feature_columns = []


    def __init__(self):

        self.feature_columns = [tf.feature_column.numeric_column(key='field' + str(i))
                                for i in range(1,65,1)]
        self.load_tensor_flow_model()

    def load_tensor_flow_model(self):
        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=self.feature_columns,
            hidden_units=[64,64,64],
            n_classes=1
        )

        pass

    # Takes the 2D designmatrix as the the features
    # and the labels are a vector of corresponding labels
    @staticmethod
    def train_input_fn(features, labels):

        features_dict = {"field" + str(i+1) : features[:,i] for i in range(0,64,1)}

        # Return the dataset.
        return (features_dict, labels)

    def train_tensor_flow_model(self, game_data):
        X, y = self.map_data(game_data)
        # self.classifier.train(
        # input_fn=lambda:self.train_input_fn(train_feature, train_label, 100),
        # steps=1000)


    # DATASET: EACH ROW, [1/0, x1,x2,...,x64].  1-> white one, 0-> black won.
    # 64 rest fields represent the board
    # game_data is 2d np array
    def map_data(self, game_data):
        y = game_data[:,0]
        X = game_data[:,1:]

        pass

    # game methods:
    def evaluate_board(self, board):
        pass
