import minishogilib
import numpy as np

import network


class Config:
    def __init__(self):
        self.batch_size = 32
        self.simulation_num = 800

        self.use_dirichlet = False
        self.dirichlet_alpha = 0.34
        self.exploration_fraction = 0.25

        self.forced_playouts = False
        self.reuse_tree = True
        self.target_pruning = False
        self.immediate = False


class MCTS():
    def __init__(self, config):
        self.config = config
        self.mcts = minishogilib.MCTS()

    def run(self, position, nn):
        root = self.mcts.set_root(position, self.config.reuse_tree)

        if self.config.immediate:
            if self.mcts.get_playouts(root) >= self.config.simulation_num:
                return root

        nninput = position.to_nninput().reshape((1, network.INPUT_CHANNEL, 5, 5))
        policy, value = nn.predict(nninput)
        value = (value + 1) / 2

        if self.config.use_dirichlet:
            moves = position.generate_moves()
            noise = np.random.gamma(self.config.dirichlet_alpha, 1, len(moves))
            frac = self.config.exploration_fraction

            for (i, m) in enumerate(moves):
                policy[0][m.to_policy_index()] = (
                    1 - frac) * policy[0][m.to_policy_index()] + frac * noise[i]

        self.mcts.evaluate(root, position, policy[0], value[0][0])

        leaf_nodes = [None for _ in range(self.config.batch_size)]
        leaf_positions = [None for _ in range(self.config.batch_size)]

        for _ in range(self.config.simulation_num // self.config.batch_size):
            if self.mcts.get_usage() > 0.9:
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
                    leaf_nodes[b], leaf_positions[b], policy[b], value[b][0])

            for b in range(self.config.batch_size):
                self.mcts.backpropagate(leaf_nodes[b], value[b][0])

        return root

    def best_move(self, node):
        return self.mcts.best_move(node)

    def dump(self, node):
        return self.mcts.dump(node, self.config.target_pruning)

    def print(self, node):
        self.mcts.print(node)

    def visualize(self, node, node_num=20):
        return self.mcts.visualize(node, node_num)
