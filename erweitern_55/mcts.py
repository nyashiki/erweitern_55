import minishogilib
import numpy as np
import threading
import time

import network


class Config:
    def __init__(self):
        self.memory_size = 1.0  # GB

        self.batch_size = 16
        self.simulation_num = 800

        self.use_dirichlet = False

        self.forced_playouts = False
        self.reuse_tree = True
        self.target_pruning = False
        self.immediate = False


class MCTS():
    def __init__(self, config):
        self.config = config
        self.mcts = minishogilib.MCTS(config.memory_size)
        self.searching = False
        self.lock = threading.Lock()

    def clear(self, shrink=False):
        self.mcts.clear(shrink)

    def run(self, position, nn, timelimit=0, verbose=False):
        self.searching = True
        start_time = time.time()

        root = self.mcts.set_root(position, self.config.reuse_tree)

        if self.config.immediate:
            if self.mcts.get_playouts(root, True) >= self.config.simulation_num:
                return root

        nninput = position.to_nninput().reshape((1, network.INPUT_CHANNEL, 5, 5))
        policy, value = nn.predict(nninput)
        value = (value + 1) / 2

        self.mcts.evaluate(
            root, position, policy[0], value[0][0], True, self.config.use_dirichlet)

        leaf_nodes = [None for _ in range(self.config.batch_size)]
        leaf_positions = [None for _ in range(self.config.batch_size)]

        loop_count = 0
        for _ in range(self.config.simulation_num // self.config.batch_size):
            if self.mcts.get_usage() > 0.9:
                break

            with self.lock:
                if not self.searching:
                    break

            current_time = time.time()
            if timelimit > 0 and (current_time - start_time) * 1000 >= timelimit:
                break

            for b in range(self.config.batch_size):
                leaf_positions[b] = position.copy(True)
                leaf_nodes[b] = self.mcts.select_leaf(
                    root, leaf_positions[b], self.config.forced_playouts)

            # use neural network to evaluate the position
            nninputs = np.zeros(
                (self.config.batch_size, network.INPUT_CHANNEL, 5, 5))
            for b in range(self.config.batch_size):
                nninputs[b] = leaf_positions[b].to_nninput().reshape(
                    (1, network.INPUT_CHANNEL, 5, 5))
            policy, value = nn.predict(nninputs)
            value = (value + 1) / 2

            for b in range(self.config.batch_size):
                value[b][0] = self.mcts.evaluate(
                    leaf_nodes[b], leaf_positions[b], policy[b], value[b][0], False, False)

            for b in range(self.config.batch_size):
                self.mcts.backpropagate(leaf_nodes[b], value[b][0])

            if verbose and loop_count % 50 == 0:
                pv_moves, q = self.mcts.info(root)
                print('info depth {} score winrate {:.3f} pv {}'.format(len(pv_moves),
                                                                        q,
                                                                        ' '.join([m.sfen() for m in pv_moves])), flush=True)

            loop_count += 1

        if verbose:
            pv_moves, q = self.mcts.info(root)
            print('info depth {} score winrate {:.3f} pv {}'.format(len(pv_moves),
                                                                    q,
                                                                    ' '.join([m.sfen() for m in pv_moves])), flush=True)

        return root

    def stop(self):
        with self.lock:
            self.searching = False

    def best_move(self, node):
        return self.mcts.best_move(node)

    def dump(self, node, remove_zeros=True):
        return self.mcts.dump(node, self.config.target_pruning, remove_zeros)

    def print(self, node):
        self.mcts.print(node)

    def debug(self, node):
        self.mcts.debug(node)

    def visualize(self, node, node_num=20):
        return self.mcts.visualize(node, node_num)
