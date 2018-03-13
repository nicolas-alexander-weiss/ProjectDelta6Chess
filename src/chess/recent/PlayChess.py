import chess
import random

import src.chess.recent.ChessAI_SKLEARN as AI
import src.chess.recent.ChessUtils as ChessUtils

import code

class PlayChess:
    def __init__(self, clf_name, train_clf=False,  own_color = "white", starting="random", guided_game = True):
        self.board = chess.Board()

        self.own_color = own_color
        self.ai_color = self.color_inverse(own_color)

        random.seed()
        if starting != "random":
            self.at_turn = starting
        elif random.randint(0,1):
            self.at_turn = "white"
        else:
            self.at_turn = "black"

        self.ai = AI.ChessAI(clf_name, train_clf)

        if guided_game:
            self.start_guided_game()

    def start_guided_game(self):
        while not self.board.is_game_over():
            self.print_board_to_file()
            print("It is", self.at_turn, "'s turn.")
            if self.at_turn == self.own_color:
                move = input("Please enter your move: ")
                # DO Move Validation first!!!
                self.board.push_san(move)
            else:
                possible_moves = self.board.legal_moves
                probability = []
                for move in possible_moves:
                    self.board.push_san(move)
                    self.ai.predict_win_prob_white(self.board)[1]
                    self.board.pop()


                i, val = max(enumerate(probability))

                self.board.push_san(possible_moves[i])
        print("GAME OVER!")




    def print_board_to_file(self):
        file = open("current_board.html", "w")
        print_this = "<html><head><meta http-equiv='refresh' content='5'></head><body>" \
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