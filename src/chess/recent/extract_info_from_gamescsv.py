import csv
import chess
import os

import src.chess.recent.ChessUtils as ChessUtils

from settings import PROJECT_ROOT


def main():
    condensed_games = []



    with open("games.csv", newline="") as csvfile:
        games = csv.reader(csvfile, delimiter=",")
        for row in games:
            condensed_games.append([row[4], row[5], row[6], row[12]])

    index = {"turns":0,"victory_status":1, "winner":2, "moves":3}

    dataset = []
    # for row in condensed_games:
    #    print (row)
        # boards = create_boards_from_moves(row["moves"])
    for game in condensed_games:
        victory_status = game[index["victory_status"]]
        if victory_status == "mate" or victory_status == "resign":
            create_boards_from_moves(game[index["moves"]])
            exit(0)




def create_boards_from_moves(moves, beg_board = 0):

    if beg_board != 0:
        return

    moves = moves.split()

    board = chess.Board()

    print(board, "\n")
    # print(moves)
    for move in moves:
        board.push_san(move)
        print(board, "\n")
        print(ChessUtils.board_to_vector(board))


    pass


#
# Program entry point
#
main()