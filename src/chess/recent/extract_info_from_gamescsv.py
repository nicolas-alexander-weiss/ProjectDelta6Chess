import csv
import chess
import numpy as np
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

    # DATASET: EACH ROW, [1/0, x1,x2,...,x64].  1-> white one, 0-> black won.
    # 64 rest fields represent the board

    dataset_mate = np.ones((1, 65))
    dataset_resign = np.ones((1, 65))
    dataset_draw = np.ones((1, 65))
    # for row in condensed_games:
    #    print (row)
        # boards = create_boards_from_moves(row["moves"])
    counter = 1
    total_rows = len(condensed_games)
    for game in condensed_games:
        victory_status = game[index["victory_status"]]
        if victory_status == "mate":
            boards = create_boards_from_moves(game[index["moves"]])
            # add winner as first element to each vector:
            if game[index["winner"]] == "white":
                boards = np.hstack((np.ones((len(boards),1)),boards))
            if game[index["winner"]] == "black":
                boards = np.hstack((np.zeros((len(boards),1)),boards))

            dataset_mate = np.vstack((dataset_mate,boards))

        elif victory_status == "resign":
            boards = create_boards_from_moves(game[index["moves"]])
            # add winner as first element to each vector:
            if game[index["winner"]] == "white":
                boards = np.hstack((np.ones((len(boards),1)),boards))
            if game[index["winner"]] == "black":
                boards = np.hstack((np.zeros((len(boards),1)),boards))

            dataset_resign = np.vstack((dataset_resign,boards))

        elif victory_status == "draw":
            boards = create_boards_from_moves(game[index["moves"]])
            # add winner as first element to each vector:
            boards = np.hstack((0.5*np.ones((len(boards), 1)), boards))

            dataset_draw = np.vstack((dataset_draw,boards))

        if counter % 1000 == 0:
            print("row: ", counter, "out of ", total_rows)
            # print(dataset_mate)
        counter = counter + 1

    np.save("boards_mate", dataset_mate)
    np.save("boards_resign", dataset_resign)
    np.save("boards_draw", dataset_draw)


def main_with_turn():
    condensed_games = []

    with open("games.csv", newline="") as csvfile:
        games = csv.reader(csvfile, delimiter=",")
        for row in games:
            condensed_games.append([row[4], row[5], row[6], row[12]])

    index = {"turns":0,"victory_status":1, "winner":2, "moves":3}

    # DATASET: EACH ROW, [1/0, x1,x2,...,x64].  1-> white one, 0-> black won.
    # 64 rest fields represent the board

    dataset_mate = np.ones((1, 66))
    dataset_resign = np.ones((1, 66))
    dataset_draw = np.ones((1, 66))
    # for row in condensed_games:
    #    print (row)
        # boards = create_boards_from_moves(row["moves"])
    counter = 1
    total_rows = len(condensed_games)
    for game in condensed_games:
        victory_status = game[index["victory_status"]]
        if victory_status == "mate":
            boards = create_boards_from_moves_with_turn(game[index["moves"]])
            # add winner as first element to each vector:
            if game[index["winner"]] == "white":
                boards = np.hstack((np.ones((len(boards),1)),boards))
            if game[index["winner"]] == "black":
                boards = np.hstack((np.zeros((len(boards),1)),boards))

            dataset_mate = np.vstack((dataset_mate,boards))

        elif victory_status == "resign":
            boards = create_boards_from_moves_with_turn(game[index["moves"]])
            # add winner as first element to each vector:
            if game[index["winner"]] == "white":
                boards = np.hstack((np.ones((len(boards),1)),boards))
            if game[index["winner"]] == "black":
                boards = np.hstack((np.zeros((len(boards),1)),boards))

            dataset_resign = np.vstack((dataset_resign,boards))

        elif victory_status == "draw":
            boards = create_boards_from_moves_with_turn(game[index["moves"]])
            # add winner as first element to each vector:
            boards = np.hstack((0.5*np.ones((len(boards), 1)), boards))

            dataset_draw = np.vstack((dataset_draw,boards))

        if counter % 1000 == 0:
            print("row: ", counter, "out of ", total_rows)
            # print(dataset_mate)
        counter = counter + 1

    np.save("boards_mate_with_turn", dataset_mate)
    np.save("boards_resign_with_turn", dataset_resign)
    np.save("boards_draw_with_turn", dataset_draw)


# returns numpy array of board positions in accordance with the given moves-string
def create_boards_from_moves_with_turn(moves, beg_board = 0):

    if beg_board != 0:
        return

    moves = moves.split()

    board = chess.Board()
    boards = np.append(np.array(ChessUtils.board_to_vector(board)),[1])

    # print(board, "\n")
    # print(moves)
    turn = False
    for move in moves:
        board.push_san(move)
        boards = np.vstack((boards,np.append(ChessUtils.board_to_vector(board),[int(turn)])))
        turn = not turn
    return boards




# returns numpy array of board positions in accordance with the given moves-string
def create_boards_from_moves(moves, beg_board = 0):

    if beg_board != 0:
        return

    moves = moves.split()

    board = chess.Board()
    boards = np.array(ChessUtils.board_to_vector(board))

    # print(board, "\n")
    # print(moves)
    for move in moves:
        board.push_san(move)
        boards = np.vstack((boards,ChessUtils.board_to_vector(board)))
    return boards


#
# Program entry point
#


if __name__ == "__main__":
    if False:
        if os.path.isfile('boards_draw.npy'):
            print("FILES ALREADY EXIST! exiting...")
            exit(1)
    main_with_turn()