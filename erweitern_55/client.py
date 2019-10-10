from optparse import OptionParser
import os
import _pickle
import socket
import sys
import threading
import utils

import mcts
import network
import selfplay


class Client:
    def __init__(self, ip, port, update=True, cpu_only=False):
        self.host = ip
        self.port = port
        self.nn = None
        self.update = update
        self.cpu_only = cpu_only

    def run(self):
        mcts_config = mcts.Config()
        mcts_config.simulation_num = 800
        mcts_config.forced_playouts = False
        mcts_config.use_dirichlet = True
        mcts_config.reuse_tree = True
        mcts_config.target_pruning = False
        mcts_config.immediate = False

        search = mcts.MCTS(mcts_config)

        self.nn = network.Network(self.cpu_only)

        while True:
            if self.update:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                    # Load neural network parameters from the server.
                    sc.connect((self.host, self.port))
                    sc.send(b'parameter')
                    data = sc.recv(16)
                    data_size = int.from_bytes(data, 'little')
                    data = utils.recvall(sc, data_size)
                    sc.send(b'parameter_ok')

                    weights = _pickle.loads(data)
                    self.nn.model.set_weights(weights)

            # Conduct selfplay.
            search.clear()
            game_record = selfplay.run(self.nn, search)

            # Send result.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                sc.connect((self.host, self.port))
                sc.send(b'record')

                data = sc.recv(1024)

                assert data == b'ready', 'Protocol violation!'

                data = _pickle.dumps(game_record, protocol=4)
                sc.send(len(data).to_bytes(16, 'little'))
                sc.sendall(data)

                sc.send(b'record_ok')


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
    (options, args) = parser.parse_args()

    client = Client(options.ip, options.port,
                    not options.no_update, options.cpu_only)
    client.run()
