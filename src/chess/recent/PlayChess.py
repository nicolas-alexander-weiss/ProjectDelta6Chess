import chess
import chess.svg


import random
import numpy as np

import src.chess.recent.ChessAI_SKLEARN as AI
import src.chess.recent.ChessUtils as ChessUtils
import operator



class PlayChess:
    def __init__(self, clf_name, train_clf=False,  own_color = "black", guided_game = True, use_65_input_v = False):
        self.board = chess.Board()#"rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
        self.use_65_input_v = use_65_input_v
        self.move_history = []

        self.own_color = own_color
        self.ai_color = self.color_inverse(own_color)

        random.seed()
        self.at_turn = "white"

        self.ai = AI.ChessAI(clf_name, train_clf)

        if guided_game:
            self.start_guided_game()

    def start_guided_game(self):
        while not self.board.is_game_over():
            self.print_board_to_file()
            print(self.board.fen())
            print("It is", self.at_turn, "'s turn.")
            if self.at_turn == self.own_color:
                move = input("Please enter your move: ")
                try:
                    if chess.Move.from_uci(move) not in self.board.legal_moves:
                        print("These are the legal moves: ", self.board.legal_moves)
                        continue
                except ValueError:
                    continue
                self.board.push_uci(move)
            else:
                probabilities = []
                move_list = []
                for move in self.board.legal_moves:
                    self.board.push_uci(str(move))
                    board_vect = ChessUtils.board_to_vector(self.board)
                    if self.use_65_input_v:
                        board_vect = np.append(ChessUtils.board_to_vector(self.board),[int(self.board.turn)])
                    board_vect = board_vect.reshape((1, -1))

                    #print(board_vect)

                    prob = self.ai.predict_win_prob_white(board_vect)[0][1]
                    # print(self.board)
                    # print(prob)

                    probabilities.append(prob)
                    move_list.append(str(move))
                    #print(move_list)
                    self.board.pop()

                i = 0
                val = 0
                if self.at_turn == "white":
                    i, val = max(enumerate(probabilities), key=operator.itemgetter(1))
                else:
                    i, val = min(enumerate(probabilities), key=operator.itemgetter(1))
                self.board.push_uci(move_list[i])
                print(self.color_inverse(self.own_color), "move: ", move_list[i])
                print("White's win prob is", probabilities[i])
            self.at_turn = self.color_inverse(self.at_turn)
        self.print_board_to_file()
        print("GAME OVER!")

    def print_board_to_file(self):
        file = open("current_board.html", "w")
        print_this = "<html><head><meta http-equiv='refresh' content='2'></head><body>" \
                     + chess.svg.board(self.board) + "</body></html>"
        file.write(print_this)
        file.close()

    def color_inverse(self, color):
        if color == "white":
            return "black"
        elif color == "black":
            return "white"
        else:
            return color


if __name__ == "__main__":
    game = PlayChess("test_clf") #, use_65_input_v=True)