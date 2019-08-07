import socket
from optparse import OptionParser
import pickle
import sys

import mcts
import network
import selfplay
import utils


class Client:
    def __init__(self, ip, port):
        self.host = ip
        self.port = port
        self.nn = network.Network()

    def run(self):
        mcts_config = mcts.Config()
        mcts_config.simulation_num = 800
        mcts_config.use_dirichlet = True

        search = mcts.MCTS(mcts_config)

        selfplay_config = selfplay.SelfplayConfig()
        selfplay_config.playout_cap_oscillation = True

        while True:
            # load neural network parameters from server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                sc.connect((self.host, self.port))
                sc.send(b'parameter')
                data = sc.recv(16)
                data_size = int.from_bytes(data, 'little')

                data = utils.recvall(sc, data_size)

                sc.send(b'parameter_ok')

                weights = pickle.loads(data)
                self.nn.model.set_weights(weights)

            # selfplay
            game_record = selfplay.run(
                self.nn, search, selfplay_config, True)

            # send result
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                sc.connect((self.host, self.port))
                sc.send(b'record')

                data = sc.recv(1024)

                assert data == b'ready', 'Protocol violation!'

                data = pickle.dumps(game_record, protocol=2)
                sc.send(len(data).to_bytes(16, 'little'))
                sc.sendall(data)

                sc.send(b'record_ok')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--ip', dest='ip',
                      help='connection target ip', default='localhost')
    parser.add_option('-p', '--port', dest='port', type='int', default=10055,
                      help='connection target port')

    (options, args) = parser.parse_args()

    client = Client(options.ip, options.port)
    client.run()
