class GameRecord():
    def __init__(self):
        self.ply = 0
        self.sfen_kif = []
        self.mcts_result = []  # (sum_N, Q, [(m, N) for m in legal_moves])
        self.learning_target_plys = []
        self.winner = 2
        self.timestamp = 0

    def to_dict(self):
        return {
            'ply': self.ply,
            'sfen_kif': self.sfen_kif,
            'mcts_result': self.mcts_result,
            'learning_target_plys': self.learning_target_plys,
            'winner': self.winner,
            'timestamp': self.timestamp
        }
