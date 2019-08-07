import pickle

import minishogilib
import numpy as np

import network


class Reservoir:
    def __init__(self):
        self.records = []
        self.learning_targets = []

    def push(self, record, minimum_playouts=0):
        index = len(self.records)

        self.records.append(record)

        for ply in record.ply:
            if record.mcts_result[ply].sum_N == 1 or record.mcts_result[ply].sum_N >= minimum_playouts:
                self.learning_targets.append((index, ply))

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
            nninputs: the representation of the neural network input layer of positions
            policies: the representation of distributions of MCTS outputs
            values: the winners of games
        """

        recent_records = self.records[-recent:]

        all_ply = 0
        for record in recent_records:
            all_ply += record.ply

        target_plys = np.random.randint(0, all_ply, mini_batch_size)
        target_plys = np.sort(target_plys)

        nninputs = np.zeros(
            (mini_batch_size, network.INPUT_CHANNEL, 5, 5), dtype='float32')
        policies = np.zeros((mini_batch_size, 69 * 5 * 5), dtype='float32')
        values = np.zeros((mini_batch_size, 1), dtype='float32')

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
                        # input
                        nninputs[target_index] = np.reshape(
                            position.to_nninput(), (network.INPUT_CHANNEL, 5, 5))

                        # policy
                        sum_N, q, playouts = record.mcts_result[ply]
                        for playout in playouts:
                            move = position.sfen_to_move(playout[0])
                            policies[target_index][move.to_policy_index()
                                                   ] = playout[1] / sum_N

                        # value
                        if record.winner == 2:
                            values[target_index] = 0
                        elif record.winner == position.get_side_to_move():
                            values[target_index] = 1
                        else:
                            values[target_index] = -1

                        target_index += 1

                        if target_index == len(target_plys) or current_ply + record.ply <= target_plys[target_index]:
                            continue_flag = False
                            break

                    if not continue_flag:
                        break

                    move = position.sfen_to_move(record.sfen_kif[ply])
                    position.do_move(move)

            current_ply += record.ply

        return nninputs, policies, values

    def len(self):
        return len(self.records)
