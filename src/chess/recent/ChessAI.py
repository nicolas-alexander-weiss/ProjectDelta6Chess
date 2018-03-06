import chess
import numpy as np


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
