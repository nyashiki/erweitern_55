import minishogilib

import math
import copy
import collections
from operator import itemgetter
import time

import numpy as np
import graphviz

from nn import network

def run_mcts(position, nn, mcts):
    root = mcts.set_root()
    nninput = position.to_nninput().reshape((1, network.INPUT_CHANNEL, 5, 5))
    policy, value = nn.predict(nninput)
    value = (value + 1) / 2

    mcts.evaluate(root, position, policy[0], value[0][0])

    SIMULATION_NUM = 800
    BATCH_SIZE = 32

    values = [None for _ in range(BATCH_SIZE)]
    leaf_nodes = [None for _ in range(BATCH_SIZE)]
    leaf_positions = [None for _ in range(BATCH_SIZE)]

    for _ in range(SIMULATION_NUM // BATCH_SIZE):
        for b in range(BATCH_SIZE):
            leaf_positions[b] = position.copy(True)
            leaf_nodes[b] = mcts.select_leaf(root, leaf_positions[b])

        # use neural network to evaluate the position
        nninputs = np.zeros((BATCH_SIZE, network.INPUT_CHANNEL, 5, 5))
        for b in range(BATCH_SIZE):
            nninputs[b] = leaf_positions[b].to_nninput().reshape((1, network.INPUT_CHANNEL, 5, 5))
        policy, value = nn.predict(nninputs)
        value = (value + 1) / 2

        for b in range(BATCH_SIZE):
            values[b] = mcts.evaluate(leaf_nodes[b], leaf_positions[b], policy[b], value[b][0])

        for b in range(BATCH_SIZE):
            mcts.backpropagate(leaf_nodes[b], values[b])

    return root

def policy_max_move(position, nn):
    moves = position.generate_moves()

    nninput = np.reshape(position.to_nninput(), (1, network.INPUT_CHANNEL, 5, 5))
    policy, value = nn.predict(nninput)

    policy_max = -1
    policy_max_move = None
    for move in moves:
        p = policy[move.to_policy_index()]
        if  p > policy_max:
            policy_max = p
            policy_max_move = move

    return policy_max_move

def value_max_move(position, nn):
    moves = position.generate_moves()

    nninputs = np.zeros((len(moves), network.INPUT_CHANNEL, 5, 5))
    for (i, move) in enumerate(moves):
        pos = position.copy(False)
        pos.do_move(move)
        nninputs[i] = np.reshape(pos.to_nninput(), (network.INPUT_CHANNEL, 5, 5))

    policy, value = nn.predict(nninputs)
    value = (value + 1) / 2

    value_max = -1
    value_max_move = None
    for (i, move) in enumerate(moves):
        v = value[i][0]
        if v > value_max:
            value_max = v
            value_max_move = move

    return value_max_move

def main():
    position = minishogilib.Position()
    position.set_start_position()

    neural_network = network.Network()
    # neural_network.load('')

    # do predict once because the first prediction takes more time than latter one
    random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
    neural_network.predict(random_input)

    mcts = minishogilib.MCTS()

    while True:
        start_time = time.time()
        root = run_mcts(position, neural_network, mcts)
        elapsed = time.time() - start_time

        best_move = mcts.best_move(root)

        if best_move.is_null_move():
            break

        position.do_move(best_move)

        print('--------------------')
        position.print()
        print(best_move)
        mcts.print(root)
        print('time:', elapsed)
        print('--------------------')

if __name__ == '__main__':
    print(minishogilib.__version__)

    # fix the seed
    np.random.seed(0)

    main()
