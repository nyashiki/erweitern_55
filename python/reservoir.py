import pickle

import minishogilib
import numpy as np
import random

import network


class Reservoir(object):
    def __init__(self):
        self.records = []
        self.learning_targets = []

    def push(self, record):
        index = len(self.records)
        self.records.append(record)
        self.learning_targets.append(record.learning_target_plys)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=2)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.__dict__.clear()
        self.__dict__.update(data)

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

        # add index
        recent_records = self.records[-recent:]
        recent_targets = self.learning_targets[-recent:]
        target_plys = [[(i, t) for t in x]
                       for (i, x) in enumerate(recent_targets)]

        # flatten targets
        target_plys = [j for i in target_plys for j in i]

        target_plys = random.sample(target_plys, mini_batch_size)
        target_plys.sort()

        nninputs = np.zeros(
            (mini_batch_size, network.INPUT_CHANNEL, 5, 5), dtype='float32')
        policies = np.zeros((mini_batch_size, 69 * 5 * 5), dtype='float32')
        values = np.zeros((mini_batch_size, 1), dtype='float32')

        target_index = 0

        while target_index < mini_batch_size:
            position = minishogilib.Position()
            position.set_start_position()

            record = recent_records[target_plys[target_index][0]]

            ply = 0
            while True:
                if ply == target_plys[target_index][1]:
                    # input
                    nninputs[target_index] = np.reshape(
                        position.to_nninput(), (network.INPUT_CHANNEL, 5, 5))

                    # policy
                    sum_N, q, playouts = record.mcts_result[target_plys[target_index][1]]
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

                    if target_index == mini_batch_size or target_plys[target_index - 1][0] != target_plys[target_index][0]:
                        break

                move = position.sfen_to_move(record.sfen_kif[ply])
                position.do_move(move)
                ply += 1

        return nninputs, policies, values

    def len(self):
        return len(self.records)

    def len_learning_targets(self):
        flatten = [j for i in self.learning_targets for j in i]
        return len(flatten)
