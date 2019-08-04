import datetime
import socket
from optparse import OptionParser
import pickle
import sys

import mcts
from nn import network
from reservoir import Reservoir
import utils

class Trainer:
    def __init__(self):
        self.reservoir = Reservoir()
        self.nn = network.Network()
        self.steps = 0

    def collect_records(self, port):
        sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sc.bind(('localhost', port))
        sc.listen(128)

        print('Ready')

        log_file = open('connection_log.txt', 'w')

        while True:
            conn, addr = sc.accept()
            message = conn.recv(1024)

            if message == b'parameter':
                data = pickle.dumps(self.nn.model.get_weights(), protocol=2)
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

                if self.reservoir.len() % 10 == 0:
                    self.reservoir.save('records.pkl')

            log_file.flush()
            conn.close()

    def update_parameters(self):
        BATCH_SIZE = 1024
        RECENT_GAMES = 50000

        while True:
            nninputs, policies, values = self.reservoir.sample(BATCH_SIZE, RECENT_GAMES)

            # Update neural network parameters
            loss = self.nn.step(nninputs, policies, values)
            print(loss)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--port', dest='port', help='port')

    trainer = Trainer()

    # Make the server which receives game records by selfplay from clients
    # trainer.collect_records()

    # Continue to update the neural network parameters
    # trainer.update_parameters()
