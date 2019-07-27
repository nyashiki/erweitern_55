import minishogilib

import math
import copy
import collections
from operator import itemgetter
import time

import numpy as np
import graphviz

from nn import network

def run_mcts(position, nn):
    mcts = minishogilib.MCTS()

    root = mcts.set_root()
    mcts.evaluate(root, position, np.random.rand(69 * 5 * 5).astype('f'), np.random.rand())

    SIMULATION_NUM = 800
    BATCH_SIZE = 8

    values = [None for _ in range(BATCH_SIZE)]
    leaf_nodes = [None for _ in range(BATCH_SIZE)]
    leaf_positions = [None for _ in range(BATCH_SIZE)]

    for _ in range(SIMULATION_NUM // BATCH_SIZE):
        for b in range(BATCH_SIZE):
            leaf_positions[b] = position.copy(False)
            leaf_nodes[b] = mcts.select_leaf(root, leaf_positions[b])

        # neural network here

        for b in range(BATCH_SIZE):
            values[b] = mcts.evaluate(leaf_nodes[b], leaf_positions[b], np.random.rand(69 * 5 * 5).astype('f'), np.random.rand())

        for b in range(BATCH_SIZE):
            mcts.backpropagate(leaf_nodes[b], values[b])

    return mcts.best_move(root)

def visualize(node, filename='search_tree'):
    search_tree = graphviz.Digraph(format='png')
    search_tree.attr('node', shape='box')

    nodes = [node]
    parents = {}
    node_count = 0

    while node_count < 20:
        if len(nodes) == 0:
            break

        max_N_index = 0
        for i in range(len(nodes)):
            if nodes[i].N > nodes[max_N_index].N:
                max_N_index = i

        max_node = nodes.pop(max_N_index)
        search_tree.node(str(node_count), 'N:{}\nP:{:.3f}\nV:{:.3f}\nQ:{:.3f}'.format(max_node.N, max_node.P, max_node.V, 0 if max_node.N == 0 else max_node.W / max_node.N))
        if max_node in parents:
            search_tree.edge(str(parents[max_node][0]), str(node_count), label=parents[max_node][1].sfen())

        for (move, child) in max_node.children.items():
            parents[child] = (node_count, move)
            nodes.append(child)

        node_count += 1

    search_tree.render(filename)

def main():
    position = minishogilib.Position()
    position.set_start_position()

    neural_network = network.Network()
    neural_network.load('./nn/weights/epoch_052.h5')

    # 1回predictを行い，1回目の実行が遅くならないようにする
    random_input = np.random.rand(1, 5, 5, 134)
    neural_network.predict(random_input)

    while True:
        start_time = time.time()
        best_move = run_mcts(position, neural_network)
        elapsed = time.time() - start_time

        if best_move.is_null_move():
            break

        position.do_move(best_move)

        print('--------------------')
        position.print()
        print('time:', elapsed)
        print('--------------------')

        # visualize(root)

    # start_time = time.time()
    # root = run_mcts(position, neural_network)
    # end_time = time.time()
    # print('elapsed', end_time - start_time)
    # for (k, v) in root.children.items():
    #     print(k, '\n  N:', v.N, 'P:', v.P, 'V:', v.V)

if __name__ == '__main__':
    # output minishogilib version
    print(minishogilib.version())

    # fix the seed
    np.random.seed(0)

    main()
