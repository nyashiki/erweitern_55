import pickle

import minishogilib
import numpy as np

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

    def sample(self, mini_batch_size, recent):
        """Sample positions from game records

        # Arguments:
            mini_batch_size: the size of array of positions
            recent: How many recent games are the target of sampling

        # Returns:
            [(position, mcts_output)]
                where
                    mcts_output: (sfen_move, N / playout_num)
        """

        recent_records = self.records[-recent:]

        all_ply = 0
        for record in recent_records:
            all_ply += record.ply

        target_plys = np.random.randint(0, all_ply, mini_batch_size)
        target_plys = np.sort(target_plys)

        mini_batch = []

        current_ply = 0
        target_index = 0

        for record in recent_records:
            if target_index == len(target_plys):
                break

            if current_ply + record.ply > target_plys[target_index]:
                position = minishogilib.Position()
                position.set_start_position()

                for ply in range(record.ply):
                    continue_flag = True

                    while current_ply + ply == target_plys[target_index]:
                        mcts_result = record.mcts_result[ply]
                        mcts_result = [(m, N / mcts_result[0]) for (m, N) in mcts_result[2]]

                        mini_batch.append((position.copy(True), mcts_result))
                        target_index += 1

                        if target_index == len(target_plys) or current_ply + record.ply <= target_plys[target_index]:
                            continue_flag = False
                            break

                    if not continue_flag:
                        break

                    move = position.sfen_to_move(record.sfen_kif[ply])
                    position.do_move(move)

            current_ply += record.ply

        return mini_batch

    def len(self):
        return len(self.records)
