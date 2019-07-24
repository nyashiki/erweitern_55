import minishogilib

import math
import copy
import collections
from operator import itemgetter
import time

import numpy as np
from nn import network

class Node:
    def __init__(self, policy=0):
        self.N = 0
        self.V = 0
        self.P = policy
        self.W = 0
        self.children = {}
        self.is_terminal = False
        self.virtual_loss = 0

    def get_puct(self, parent_N):
        c_base = 19652
        c_init = 1.25

        C = ((1 + self.N + c_base) / c_base) + c_init

        Q = 0 if self.N == 0 else (1 - self.W / (self.N + self.virtual_loss))
        U = C * self.P * math.sqrt(parent_N) / (1 + self.N + self.virtual_loss)

        return (Q + U)

    def expanded(self):
        return len(self.children) > 0 and not self.is_terminal

def select_child(node):
    _, move, child = max(((child.get_puct(node.N), move, child) for move, child in node.children.items()), key=itemgetter(0))

    return move, child

def evaluate(nodes, positions, nn):
    nn_inputs = np.zeros((len(nodes), 68, 5, 5))

    for b in range(len(nodes)):
        nn_input = np.array(positions[b].to_nninput()).reshape(1, 68, 5, 5)
        nn_inputs[b] = nn_input

    # Use the neural network to predict current win rate and probabilities along the next moves
    nn_inputs = np.transpose(nn_inputs, axes=[0, 2, 3, 1])
    policies, values = nn.predict(nn_inputs)

    values = values.flatten()
    values = (values + 1) / 2

    policies = policies.reshape(len(nodes), 5, 5, 69)
    policies = np.transpose(policies, [0, 3, 1, 2])

    for b in range(len(nodes)):
        node = nodes[b]
        position = positions[b]

        moves = position.generate_moves()

        policy = policies[b]
        legal_policy_sum = np.sum([policy[network.move_to_policy_index(position.get_side_to_move(), m)] for m in moves])

        is_repetition, is_check_repetition = position.is_repetition()
        if is_repetition or len(moves) == 0:
            node.is_terminal = True

        if node.is_terminal:
            values[b] = 0

        # sef value and policy
        node.V = values[b]

        for i, move in enumerate(moves):
            node.children[move] = Node(policy[network.move_to_policy_index(position.get_side_to_move(), move)] / legal_policy_sum)

    return values

def backpropagate(search_path, value):
    flip = False
    while True:
        node = search_path.pop()

        node.W += value if not flip else (1 - value)
        node.N += 1

        if len(search_path) == 0:
            break

        node.virtual_loss -= 1
        flip = not flip

def run_mcts(position, nn):
    root = Node()
    evaluate([root], [position], nn)

    SIMULATION_NUM = 800
    BATCH_SIZE = 16

    search_paths = [None for _ in range(BATCH_SIZE)]
    leaf_nodes = [None for _ in range(BATCH_SIZE)]
    leaf_positions = [None for _ in range(BATCH_SIZE)]

    for _ in range(SIMULATION_NUM // BATCH_SIZE):
        for b in range(BATCH_SIZE):
            leaf_positions[b] = position.copy(False)

            node = root
            search_paths[b] = collections.deque([node])

            while node.expanded():
                move, node = select_child(node)
                node.virtual_loss += 1

                leaf_positions[b].do_move(move)

                # record path
                search_paths[b].append(node)

            leaf_nodes[b] = node

        values = evaluate(leaf_nodes, leaf_positions, nn)

        for b in range(BATCH_SIZE):
            backpropagate(search_paths[b], values[b])

    return root


def main():
    position = minishogilib.Position()
    position.set_start_position()

    neural_network = network.Network()
    neural_network.load('./nn/weights/epoch_99.h5')

    start_time = time.time()
    root = run_mcts(position, neural_network)
    end_time = time.time()

    print('elapsed', end_time - start_time)

    for (k, v) in root.children.items():
        print(k, '\n  N:', v.N, 'P:', v.P, 'V:', v.V)

if __name__ == '__main__':
    # output minishogilib version
    print(minishogilib.version())

    # fix the seed
    np.random.seed(0)

    main()
