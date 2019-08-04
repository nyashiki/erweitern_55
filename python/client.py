import socket
import pickle
import sys

import mcts
from nn import network
import selfplay
import utils

if __name__ == '__main__':
    nn = network.Network()
    search = mcts.MCTS(mcts.Config())
    selfplay_config = selfplay.SelfplayConfig()
    selfplay_config.use_dirichlet = True

    host = 'localhost'
    port = 9000

    while True:
        # load neural network parameters from server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
            sc.connect((host, port))
            sc.send(b'parameter')
            data = sc.recv(16)
            data_size = int.from_bytes(data, 'little')
            data = utils.recvall(sc, data_size)

            weights = pickle.loads(data)
            nn.model.set_weights(weights)

        # selfplay
        game_record = selfplay.run(nn, search, selfplay_config)

        # send result
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sc:
            sc.connect((host, port))
            sc.send(b'record')

            data = pickle.dumps(game_record, protocol=2)
            sc.send(sys.getsizeof(data).to_bytes(16, 'little'))
            sc.send(data)
