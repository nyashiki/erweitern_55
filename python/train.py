import datetime
import socket
from optparse import OptionParser
import pickle
import sys
import threading

import mcts
from nn import network
from reservoir import Reservoir
import tensorflow as tf
import tensorflow.keras.backend as K
import utils

class Trainer():
    def __init__(self, port):
        self.port = port

        self.reservoir = Reservoir()
        self.nn = network.Network()
        self.nn.model._make_predict_function()
        self.session = K.get_session()
        self.graph = tf.compat.v1.get_default_graph()

        self.steps = 0

        self.nn_lock = threading.Lock()
        self.reservoir_lock = threading.Lock()

        # self.nn.load('./nn/weights/epoch_030.h5')
        # self.reservoir.load('records.pkl')

    def collect_records(self):
        sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sc.bind(('localhost', self.port))
        sc.listen(128)

        print('Ready')

        log_file = open('connection_log.txt', 'w')

        while True:
            conn, addr = sc.accept()
            message = conn.recv(1024)

            if message == b'parameter':
                with self.nn_lock:
                    with self.session.as_default():
                        with self.graph.as_default():
                            data = pickle.dumps(self.nn.model.get_weights(), protocol=2)
                            conn.send(sys.getsizeof(data).to_bytes(16, 'little'))
                            conn.send(data)

                log_file.write('[{}] sent the parameters to {}\n'.format(datetime.datetime.now(datetime.timezone.utc), str(addr)))

            elif message == b'record':
                data = conn.recv(16)
                data_size = int.from_bytes(data, 'little')

                data = utils.recvall(conn, data_size)

                with self.reservoir_lock:
                    game_record = pickle.loads(data)
                    self.reservoir.push(game_record)

                log_file.write('[{}] received a game record from {}\n'.format(datetime.datetime.now(datetime.timezone.utc), str(addr)))

                if self.reservoir.len() % 10 == 0:
                    with reservoir_lock:
                        self.reservoir.save('records.pkl')

            log_file.flush()
            conn.close()

    def update_parameters(self):
        BATCH_SIZE = 1024
        RECENT_GAMES = 50000

        log_file = open('training_log.txt', 'w')

        while True:
            with self.reservoir_lock:
                nninputs, policies, values = self.reservoir.sample(BATCH_SIZE, RECENT_GAMES)

            # Update neural network parameters
            with self.nn_lock:
                with self.session.as_default():
                    with self.graph.as_default():
                        loss_sum, policy_loss, value_loss = self.nn.step(nninputs, policies, values)

            log_file.write('{}, {}, {}, {}\n'.format(datetime.datetime.now(datetime.timezone.utc), str(loss_sum), str(policy_loss), str(value_loss)))
            log_file.flush()

    def run(self):
        # Make the server which receives game records by selfplay from clients
        collect_records_thread = threading.Thread(target=self.collect_records)

        # Continue to update the neural network parameters
        update_parameters_thread = threading.Thread(target=self.update_parameters)

        collect_records_thread.start()
        update_parameters_thread.start()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--port', dest='port', help='port')

    (options, args) = parser.parse_args()

    trainer = Trainer(int(options.port))
    trainer.run()
