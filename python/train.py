import datetime
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

    log_file = open('connection_log.txt', 'w')

    while True:
        conn, addr = sc.accept()
        message = conn.recv(1024)

        if message == b'parameter':
            data = pickle.dumps(nn.model.get_weights(), protocol=2)
            conn.send(sys.getsizeof(data).to_bytes(16, 'little'))
            conn.send(data)

            log_file.write('[{}] sent the parameters to {}\n'.format(datetime.datetime.now(datetime.timezone.utc), str(addr)))

        elif message == b'record':
            data = conn.recv(16)
            data_size = int.from_bytes(data, 'little')

            data = utils.recvall(conn, data_size)

            game_record = pickle.loads(data)
            reservoir.push(game_record)

            log_file.write('[{}] received a game record from {}\n'.format(datetime.datetime.now(datetime.timezone.utc), str(addr)))

            if reservoir.len() % 10 == 0:
                reservoir.save('records.pkl')

        log_file.flush()
        conn.close()
