class GameRecord:
    def __init__(self):
        self.ply = 0
        self.sfen_kif = []
        self.mcts_result = []  # (sum_N, Q, [for (m, N) for m in legal_moves])
        self.winner = None
        self.timestamp = 0
