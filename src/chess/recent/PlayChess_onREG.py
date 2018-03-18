
import numpy as np

import chess
import chess.svg

import src.chess.recent.NN_Regressor as AI
import src.chess.recent.ChessUtils as ChessUtils

color = {True: "white", False:" black"}
mltpl = {True: 1, "white": 1, False: -1, "black": -1}

# labels * -1 if black wins
labels = {"draw": 0, "resign": 0.75, "mate": 1}

# X structure:
# castling indicator if still possible as true/false -> 1/0
# left/right from white's perspective
# [board(64 elements),
#   cstl_white_left, cstl_white_right, cstl_black_left, cstl_black_right,
#   en_passant_possible,
#   at_turn]

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
n_features = len(training_data_beg)


class PlayChess:
    def __init__(self, reg_name, train_reg=True, verbose=True):
        self.ai = AI.NN_Regressor(reg_name, train_reg)

        self.train = train_reg
        self.verbose = verbose

        self.acc_training_data = None
        self.acc_labels = None
        self.board = chess.Board()

    def comp_vs_comp(self):
        self.board = chess.Board()

        self.acc_training_data = training_data_beg

        # play game first
        while not self.board.is_game_over():

            # print("at turn:", color[self.board.turn])
            white_at_turn = self.board.turn

            possible_moves = list(self.board.legal_moves)

            self.board.push(possible_moves[0])
            best_move = possible_moves[0]
            best_val = self.ai.clf.predict([self.get_feature_vector(self.board)])[0]
            if self.board.is_checkmate():
                best_val = mltpl[white_at_turn] * 1000000
            elif self.board.can_claim_draw():
                best_val = mltpl[not self.board.turn] * -1000000

            self.board.pop()

            for i in range(1, len(possible_moves),1):
                self.board.push(possible_moves[i])
                feature_vector = self.get_feature_vector(self.board)
                val = self.ai.clf.predict([feature_vector])[0]
                #print(val)
                if self.board.is_checkmate():
                    val = mltpl[white_at_turn] * 1000000
                elif self.board.can_claim_draw():
                    val = mltpl[not self.board.turn] * -1000000

                if white_at_turn and val > best_val:
                    best_val = val
                    best_move = possible_moves[i]
                elif (not white_at_turn) and val < best_val:
                    best_val = val
                    best_move = possible_moves[i]
                self.board.pop()
                # print(val)

            if white_at_turn and best_val < 0 and self.board.can_claim_draw():
                break
            elif (not white_at_turn) and best_val > 0 and self.board.can_claim_draw():
                break

            self.acc_training_data = np.vstack((self.acc_training_data, self.get_feature_vector(self.board)))
            self.board.push(best_move)
            # print("chosen: ", best_val)

            if self.verbose:
                self.print_board_to_file()
                input("continue? -> enter")

        self.acc_labels = np.ones((len(self.acc_training_data), 1)) * self.get_label(self.board)
        if self.board.is_checkmate():
            print("GameOver, Checkmate, Winner:", not self.board.turn)
        else:
            print("GameOver, draw")
        print("rounds:", self.board.fullmove_number, "board:\n", self.board.fen())



        # training
        if self.train:
            self.ai.train_clf(self.acc_training_data, self.acc_labels)

    def get_label(self, board):
        # board = chess.Board(board)
        if board.is_checkmate():
            return labels["mate"] * mltpl[not board.turn]
        return 0

    def get_feature_vector(self, board):

        feature_vector = np.zeros(n_features)

        for i in range(0, 64, 1):
            piece = board.piece_at(i)
            if piece is None:
                continue
            feature_vector[i] = ChessUtils.sym_to_int[piece.symbol()]

        feature_vector[64] = board.has_queenside_castling_rights(True)
        feature_vector[65] = board.has_kingside_castling_rights(True)
        feature_vector[66] = board.has_queenside_castling_rights(False)
        feature_vector[67] = board.has_kingside_castling_rights(False)
        feature_vector[68] = board.has_legal_en_passant()
        feature_vector[69] = board.turn
        return feature_vector

    def print_board_to_file(self):
        file = open("current_board.html", "w")
        print_this = "<html><head><meta http-equiv='refresh' content='2'></head><body>" \
                     + chess.svg.board(self.board) + "</body></html>"
        file.write(print_this)
        file.close()

if __name__ == "__main__":
    game = PlayChess("reg70_0.2", verbose=False)
    for i in range(0,100,1):
        game.comp_vs_comp()