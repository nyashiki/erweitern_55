class GameRecord:
    def __init__(self):
        self.ply = 0
        self.sfen_kif = []
        self.mcts_result = []  # (sum_N, Q, [(m, N) for m in legal_moves])
        self.learning_target_plys = []
        self.winner = 2
        self.timestamp = 0
