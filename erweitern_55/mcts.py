import minishogilib
import numpy as np
import threading
import time

import network


class Config:
    def __init__(self):
        self.memory_size = 0.2 # GB

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

    def clear(self):
        self.mcts.clear()

    def run(self, position, nn, timelimit=0, verbose=False):
        """Run Monte-Carlo Tree search at the given position.

        # Arguments:
            position: the position at which search the next move.
            nn: Network instance.
            timelimit: the timelimit for searching (milliseconds).
            verbose: whether output logs to the standard output.

        # Returns:
            root: the root node that corresponds the given position.
        """
        self.searching = True
        start_time = time.time()

        # Step 1: Set the root node.
        root = self.mcts.set_root(position, self.config.reuse_tree)

        if self.config.immediate:
            # If the number of visit at the root node exceeds the certain number,
            # don't conduct search and return immediately.
            if self.mcts.get_playouts(root, True) >= self.config.simulation_num:
                return root

        # Step 2: Evaluate the root node.
        nninput = nn.get_input(position, True)
        policy, value = nn.predict(nninput)
        value = (value + 1) / 2
        self.mcts.evaluate(
            root, position, policy[0], value[0][0], self.config.use_dirichlet)

        # Step 3: Start searching.
        # Main loop of the Monte-Carlo tree search.
        loop_count = 0
        for _ in range(self.config.simulation_num // self.config.batch_size):
            # If the memory is used over 90%, suspend the search.
            if self.mcts.get_usage() > 0.9:
                break

            # If searching flag is False, suspend the search.
            with self.lock:
                if not self.searching:
                    break

            # Check the timelimit.
            current_time = time.time()
            if timelimit > 0 and (current_time - start_time) * 1000 >= timelimit:
                break

            leaf_nodes = [None for _ in range(self.config.batch_size)]
            leaf_positions = [position.copy(True) for _ in range(self.config.batch_size)]

            # MCTS Step 1: select leaf nodes.
            for b in range(self.config.batch_size):
                leaf_nodes[b] = self.mcts.select_leaf(
                    root, leaf_positions[b], self.config.forced_playouts)

            # MCTS Step 2: expand children of the leaf nodes and evaluate the leaf nodes.
            nninputs = nn.get_inputs(leaf_positions)
            policy, value = nn.predict(nninputs)
            value = (value + 1) / 2
            for b in range(self.config.batch_size):
                value[b][0] = self.mcts.evaluate(
                    leaf_nodes[b], leaf_positions[b], policy[b], value[b][0], False)

            # MCTS Step 3: backpropage values of the leaf nodes from the leaf nodes to the root node.
            for b in range(self.config.batch_size):
                self.mcts.backpropagate(leaf_nodes[b], value[b][0])

            # Output log.
            if verbose and loop_count % 50 == 0:
                pv_moves, q = self.mcts.info(root)
                print('info depth {} nodes {} hashfull {} score winrate {:.3f} pv {}'.format(len(pv_moves),
                                                                                             self.mcts.get_nodes(),
                                                                                             int(self.mcts.get_usage() * 1000),
                                                                                             q,
                                                                                             ' '.join([m.sfen() for m in pv_moves])), flush=True)

            loop_count += 1

        # Output log.
        if verbose:
            pv_moves, q = self.mcts.info(root)
            print('info depth {} nodes {} hashfull {} score winrate {:.3f} pv {}'.format(len(pv_moves),
                                                                                         self.mcts.get_nodes(),
                                                                                         int(self.mcts.get_usage() * 1000),
                                                                                         q,
                                                                                         ' '.join([m.sfen() for m in pv_moves])), flush=True)

        return root

    def stop(self):
        """Stop searching.
        """
        with self.lock:
            self.searching = False

    def best_move(self, node):
        """Get the next move at the given node.

        # Arguments:
            node: the target node.

        # Returns:
            The next move.
        """
        return self.mcts.best_move(node)

    def softmax_sample(self, node, temperature):
        """Get the next move at the given node along softmax sampling of visit counts.

        # Arguments:
            node: the target node.

        # Returns:
            The next move.
        """
        return self.mcts.softmax_sample(node, temperature)

    def dump(self, node, remove_zeros=True):
        """Get (move, the number of visit) pairs.

        # Arguments:
            node: the target node.
            remove_zeros: ignore child nodes whose number of visit is 0.

        # Returns:
            List of (move, the number of visit) pair.
        """
        return self.mcts.dump(node, self.config.target_pruning, remove_zeros)

    def print(self, node):
        self.mcts.print(node)

    def debug(self, node):
        self.mcts.debug(node)

    def visualize(self, node, node_num=20):
        """Get the game-tree written in dot language.

        # Arguments:
            node: the target node.
            node_num: the number of nodes.

        # Returns:
            String of dot language represenattion of the game tree.
        """
        return self.mcts.visualize(node, node_num)
