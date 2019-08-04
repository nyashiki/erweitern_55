import socket
from optparse import OptionParser
import pickle
import sys

import mcts
from nn import network
import selfplay
import utils


class Client:
    def __init__(self, ip, port):
        self.host = ip
        self.port = int(port)
        self.nn = network.Network()

    def run(self):
        selfplay_config = selfplay.SelfplayConfig()
        search = mcts.MCTS(mcts.Config())
        selfplay_config.use_dirichlet = True

        while True:
            # load neural network parameters from server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                sc.connect((self.host, self.port))
                sc.send(b'parameter')
                data = sc.recv(16)
                data_size = int.from_bytes(data, 'little')
                data = utils.recvall(sc, data_size)

                weights = pickle.loads(data)
                self.nn.model.set_weights(weights)

            # selfplay
            game_record = selfplay.run(self.nn, search, selfplay_config)

            # send result
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
                sc.connect((self.host, self.port))
                sc.send(b'record')

                data = pickle.dumps(game_record, protocol=2)
                sc.send(sys.getsizeof(data).to_bytes(16, 'little'))
                sc.send(data)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-i', '--ip', dest='ip', help='connection target ip')
    parser.add_option('-p', '--port', dest='port', help='connection target port')

    (options, args) = parser.parse_args()

    client = Client(options.ip, options.port)
    client.run()
