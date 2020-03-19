from datetime import datetime
import minishogilib
import numpy as np
from optparse import OptionParser
import os
import _pickle
import sys
import threading
import time
import urllib.request

import gamerecord
import network


class Client:
    """Client that connects the server and conducts selfplay games.
    """

    def __init__(self, ip, port, update=True, cpu_only=False, checkmate_depth=7, worker_num=1):
        self.host = ip
        self.port = port
        self.nn = None
        self.update = update
        self.cpu_only = cpu_only
        self.checkmate_depth = checkmate_depth
        self.worker_num = worker_num

    def run(self):
        memory_size = 0.01
        simulation_num = 20

        device = 'cpu' if self.cpu_only else 'gpu'
        self.nn = network.Network(device)

        positions = [minishogilib.Position() for _ in range(self.worker_num)]
        searchs = [minishogilib.MCTS(memory_size) for _ in range(self.worker_num)]
        game_records = [gamerecord.GameRecord() for _ in range(self.worker_num)]
        ins = np.zeros([self.worker_num] + self.nn.input_shape)

        for position in positions:
            position.set_start_position()

        while True:
            # Ask the server the current neural network parameters.
            if self.update:
                url = 'http://{}:{}/weight'.format(self.host, self.port)
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as res:
                    weights = _pickle.loads(res.read())
                    self.nn.model.set_weights(weights)

            # MCTS begin.
            # Step 1: Set the root nodes.
            roots = [None for _ in range(self.worker_num)]
            for (i, position) in enumerate(positions):
                root = searchs[i].set_root(position, True)
                roots[i] = root

                if not searchs[i].expanded(root):
                    ins[i] = self.nn.get_input(position)

            # Step 2: Evaluate the root nodes.
            policy, value = self.nn.predict(ins)
            value = (value + 1) / 2
            for (i, position) in enumerate(positions):
                searchs[i].evaluate(roots[i], position, policy[i], value[i][0])

                # Add dirichlet noise.
                searchs[i].add_noise(roots[i])

            for _ in range(simulation_num):
                leaf_nodes = [None for _ in range(self.worker_num)]
                leaf_positions = [positions[i].copy(True) for i in range(self.worker_num)]

                # Select leaf nodes.
                for i in range(self.worker_num):
                    leaf_nodes[i] = searchs[i].select_leaf(roots[i], leaf_positions[i], False)
                    ins[i] = self.nn.get_input(leaf_positions[i])

                # Evaluate leaf nodes and backpropagate values.
                policy, value = self.nn.predict(ins)
                value = (value + 1) / 2
                for i in range(self.worker_num):
                    searchs[i].evaluate(leaf_nodes[i], leaf_positions[i], policy[i], value[i][0])
                    searchs[i].backpropagate(leaf_nodes[i])

            # Do the next move.
            for (i, position) in enumerate(positions):
                if position.get_ply() < 30:
                    next_move = searchs[i].softmax_sample(roots[i], temperature=1.0)
                else:
                    next_move = searchs[i].best_move(roots[i])

                position.do_move(next_move)

                # Record for the training.
                game_records[i].sfen_kif.append(next_move.sfen())
                game_records[i].mcts_result.append(searchs[i].dump(roots[i], False, True))
                game_records[i].learning_target_plys.append(game_records[i].ply)
                game_records[i].ply += 1

                # Judge whether game is over or not.
                game_over = False
                is_repetition, my_check_repetition, op_check_repetition = position.is_repetition()
                if my_check_repetition:
                    game_records[i].winner = 1 - position.get_side_to_move()
                    game_over = True
                elif op_check_repetition:
                    game_records[i].winner = position.get_side_to_move()
                    game_over = True
                elif is_repetition:
                    game_records[i].winner = 1
                    game_over = True
                elif position.get_ply() >= 320:
                    game_records[i].winner = 2
                    game_over = True

                moves = position.generate_moves()
                if len(moves) == 0:
                    game_records[i].winner = 1 - position.get_side_to_move()
                    game_over = True

                # If end, send the result.
                if game_over:
                    game_records[i].timestamp = int(datetime.now().timestamp())
                    url = 'http://{}:{}/record'.format(self.host, self.port)
                    data = _pickle.dumps(game_records[i], protocol=4)
                    req = urllib.request.Request(url, data)
                    with urllib.request.urlopen(req) as res:
                        pass

                    # Set up for the next game.
                    position.set_start_position()
                    searchs[i].clear()
                    game_records[i] = gamerecord.GameRecord()

            # MCTS end.




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--ip', dest='ip',
                      help='connection target ip', default='localhost')
    parser.add_option('-p', '--port', dest='port', type='int', default=10055,
                      help='connection target port')
    parser.add_option('-s', '--no-update', action='store_true', dest='no_update', default=False,
                      help='If true, neural network parameters will not be updated.')
    parser.add_option('-c', '--cpu', action='store_true', dest='cpu_only', default=False,
                      help='If true, use CPU only.')
    parser.add_option('-d', '--checkmate-depth', dest='checkmate_depth',
                      default=7, help='The depth of checkmate searching.')
    parser.add_option('-w', '--worker-num', type='int', dest='worker_num',
                      default=1, help='The number of workers.')

    (options, args) = parser.parse_args()

    client = Client(options.ip, options.port,
                    not options.no_update, options.cpu_only, options.checkmate_depth, options.worker_num)
    client.run()
