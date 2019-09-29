import socket
from optparse import OptionParser
import _pickle
import sys

import mcts
import network
import selfplay
import utils


class Client:
    def __init__(self, ip, port, update=True, update_iter=10):
        self.host = ip
        self.port = port
        self.nn = None
        self.update = update
        self.update_iter = update_iter

    def run(self):
        mcts_config = mcts.Config()
        mcts_config.simulation_num = 800
        mcts_config.forced_playouts = False
        mcts_config.use_dirichlet = True
        mcts_config.reuse_tree = True
        mcts_config.target_pruning = False
        mcts_config.immediate = False

        search = mcts.MCTS(mcts_config)

        iter = 0

        while True:
            # load neural network parameters from server
            if self.nn is None or (self.update and iter % self.update_iter == 0):
                if self.nn is None:
                    self.nn = network.Network()

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                    sc.connect((self.host, self.port))
                    sc.send(b'parameter')
                    data = sc.recv(16)
                    data_size = int.from_bytes(data, 'little')

                    data = utils.recvall(sc, data_size)

                    sc.send(b'parameter_ok')

                    weights = _pickle.loads(data)
                    self.nn.model.set_weights(weights)

            # selfplay
            search.clear()
            game_record = selfplay.run(self.nn, search)

            # send result
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                sc.connect((self.host, self.port))
                sc.send(b'record')

                data = sc.recv(1024)

                assert data == b'ready', 'Protocol violation!'

                data = _pickle.dumps(game_record, protocol=2)
                sc.send(len(data).to_bytes(16, 'little'))
                sc.sendall(data)

                sc.send(b'record_ok')

            iter += 1


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--ip', dest='ip',
                      help='connection target ip', default='localhost')
    parser.add_option('-p', '--port', dest='port', type='int', default=10055,
                      help='connection target port')
    parser.add_option('-s', '--no-update', action='store_true', dest='no_update', default=False,
                      help='If true, neural network parameters will not be updated.',)
    parser.add_option('-u', '--update-iter', dest='update_iter', type='int',
                      default=10, help='The iteration to update neural network parameters.')

    (options, args) = parser.parse_args()

    client = Client(options.ip, options.port,
                    not options.no_update, options.update_iter)
    client.run()
