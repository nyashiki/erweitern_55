import minishogilib
import numpy as np

from nn import network
import search

class Config:
    def __init__(self):
        self.batch_size = 32
        self.simulation_num = 800

class MCTS(search.Search):
    def __init__(self, config):
        self.config = config
        self.mcts = minishogilib.MCTS()

    def run(self, position, nn):
        root = self.mcts.set_root()
        nninput = position.to_nninput().reshape((1, network.INPUT_CHANNEL, 5, 5))
        policy, value = nn.predict(nninput)
        value = (value + 1) / 2

        self.mcts.evaluate(root, position, policy[0], value[0][0])

        values = [None for _ in range(self.config.batch_size)]
        leaf_nodes = [None for _ in range(self.config.batch_size)]
        leaf_positions = [None for _ in range(self.config.batch_size)]

        for _ in range(self.config.simulation_num // self.config.batch_size):
            for b in range(self.config.batch_size):
                leaf_positions[b] = position.copy(True)
                leaf_nodes[b] = self.mcts.select_leaf(root, leaf_positions[b])

            # use neural network to evaluate the position
            nninputs = np.zeros((self.config.batch_size, network.INPUT_CHANNEL, 5, 5))
            for b in range(self.config.batch_size):
                nninputs[b] = leaf_positions[b].to_nninput().reshape((1, network.INPUT_CHANNEL, 5, 5))
            policy, value = nn.predict(nninputs)
            value = (value + 1) / 2

            for b in range(self.config.batch_size):
                values[b] = self.mcts.evaluate(leaf_nodes[b], leaf_positions[b], policy[b], value[b][0])

            for b in range(self.config.batch_size):
                self.mcts.backpropagate(leaf_nodes[b], values[b])

        return root

    def best_move(self, node):
        return self.mcts.best_move(node)

    def dump(self, node):
        self.mcts.print(node)
