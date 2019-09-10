import bisect
import minishogilib
from joblib import Parallel, delayed
import numpy as np
import os
import random
import simplejson

import gamerecord
import network

def _get_datasets(record, target_ply, mcts_result):
    position = minishogilib.Position()
    position.set_start_position()
    ply = 0

    kif = record.sfen_kif

    for ply in range(target_ply):
        move = position.sfen_to_move(kif[ply])
        position.do_move(move)

    # input
    nn_input = np.reshape(position.to_nninput(), (network.INPUT_CHANNEL, 5, 5))

    # policy
    policy = np.zeros((69 * 5 * 5), dtype='float32')
    sum_N, q, playouts = mcts_result
    for playout in playouts:
        move = position.sfen_to_move(playout[0])
        policy[move.to_policy_index()] = playout[1] / sum_N

    # value
    if record.winner == 2:
        value = 0
    elif record.winner == position.get_side_to_move():
        value = 1
    else:
        value = -1

    return nn_input, policy, value

class Reservoir(object):
    def __init__(self, json_dump='records.json'):
        self.records = []
        self.learning_targets = []
        self.json_dump = json_dump

        if os.path.isfile(json_dump):
            self._load()

    def _load(self):
        with open(self.json_dump, 'r') as f:
            line = f.readline()

            while line:
                data = simplejson.loads(line)

                record = gamerecord.GameRecord()
                record.ply = data['ply']
                record.sfen_kif = data['sfen_kif']
                record.mcts_result = data['mcts_result']
                record.learning_target_plys = data['learning_target_plys']
                record.winner = data['winner']
                record.timestamp = data['timestamp']

                self.records.append(record)

                line = f.readline()

        self.learning_targets = []
        for record in self.records:
            self.learning_targets.append(record.learning_target_plys)

    def push(self, record):
        self.records.append(record)
        self.learning_targets.append(record.learning_target_plys)

        with open(self.json_dump, 'a') as f:
            simplejson.dump(record.to_dict(), f)
            f.write('\n')

    def sample(self, mini_batch_size, recent, discard=True):
        """Sample positions from game records

        # Arguments:
            mini_batch_size: the size of array of positions
            recent: How many recent games are the target of sampling

        # Returns:
            nninputs: the representation of the neural network input layer of positions
            policies: the representation of distributions of MCTS outputs
            values: the winners of games
        """

        if discard:
            self.records = self.records[-recent:]
            self.learning_targets = self.learning_targets[-recent:]

        # add index
        recent_records = self.records[-recent:]
        recent_targets = self.learning_targets[-recent:]
        cumulative_plys = [0 for _ in range(recent + 1)]
        for i in range(recent):
            cumulative_plys[i + 1] = cumulative_plys[i] + len(recent_targets[i])
        indicies = random.sample(range(cumulative_plys[recent]), mini_batch_size)
        indicies.sort()
        target_plys = [None for _ in range(mini_batch_size)]
        lo = 0
        for i in range(mini_batch_size):
            index = bisect.bisect_right(cumulative_plys, indicies[i], lo=lo) - 1
            ply = recent_targets[index][indicies[i] - cumulative_plys[index]]
            target_plys[i] = (index, ply)
            lo = index

        datasets = Parallel(n_jobs=-1)([delayed(_get_datasets)(recent_records[t[0]], t[1], recent_records[t[0]].mcts_result[t[1]]) for t in target_plys])

        nninputs = np.array([x[0] for x in datasets], dtype='float')
        policies = np.array([x[1] for x in datasets], dtype='float')
        values = np.array([x[2] for x in datasets], dtype='float')

        return nninputs, policies, values

    def len(self):
        return len(self.records)

    def len_learning_targets(self):
        flatten = [j for i in self.learning_targets for j in i]
        return len(flatten)
