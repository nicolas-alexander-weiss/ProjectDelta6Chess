import src.chess.recent.PlayChess_onREG as play

if __name__ == "__main__":

    game = play.PlayChess("reg70_0.3_normalized_v2", verbose=True)
    for i in range(0, 1, 1):
        game.comp_vs_human()
    print("Trained on a total of", game.ai.stats[0], "board positions")
    print("Num iterations:", game.ai.clf.n_iter_)