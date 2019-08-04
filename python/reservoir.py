import pickle

class Reservoir:
    def __init__(self):
        self.records = []

    def push(self, record):
        self.records.append(record)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.records, f, protocol=2)

    def load(self, path):
        with open(path, 'rb') as f:
            self.records = pickle.load(f)

    def sample(self, mini_batch_size):
        """Sample positions from game records

        Return: [(position, mcts_output)]
            where
                mcts_output: (sfen_move, N / playout_num)
        """
        pass  # ToDo

    def len(self):
        return len(self.records)
