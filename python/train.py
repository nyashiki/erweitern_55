import socket
import pickle
import sys

import mcts
from nn import network
from reservoir import Reservoir
import utils

def run():
    reservoir = Reservoir()
    nn = network.Network()
    nn.load('./nn/weights/epoch_030.h5')

    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.bind(('localhost', 9000))
    sc.listen(128)

    print('Ready')

    while True:
        conn, addr = sc.accept()
        print('connected by', addr)

        message = conn.recv(1024)

        if message == b'parameter':
            data = pickle.dumps(nn.model.get_weights(), protocol=2)
            conn.send(sys.getsizeof(data).to_bytes(16, 'little'))
            conn.send(data)

        elif message == b'record':
            data = conn.recv(16)
            data_size = int.from_bytes(data, 'little')
            print('data_size', data_size)

            data = utils.recvall(conn, data_size)

            game_record = pickle.loads(data)
            reservoir.push(game_record)

            print(game_record.sfen_kif)

        else:
            print('Unknown command:', message)

        conn.close()
