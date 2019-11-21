import datetime
import http.server
import minishogilib
import numpy as np
from optparse import OptionParser
import os
import _pickle
import queue
import simplejson
import socketserver
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import threading
import time

import mcts
import network


class Trainer():
    """Server that collects game records sent by clients and updates neural network parameters.
    """

    def __init__(self, port, store_only=False, record_file='records.json', weight_file=None, update_record_num=0):
        RECENT_GAMES = 100000

        self.port = port

        self.reservoir = minishogilib.Reservoir(record_file, RECENT_GAMES)
        self.nn = network.Network('gpu')

        self.reservoir_lock = threading.Lock()
        self.nn_lock = threading.Lock()

        self.checkpoint_weights = self.nn.get_weights()

        self.store_only = store_only

        if record_file is not None:
            if os.path.isfile(record_file):
                self.reservoir.load(record_file)

        if weight_file is not None:
            self.nn.load(weight_file)
        else:
            self.nn.save('./weights/iter_0.h5')

        self.training_data = queue.Queue(maxsize=1)

        self.update_record_num = update_record_num
        self.new_record_count = [0]
        self.new_record_count_lock = threading.Lock()

    def _sample_datasets(self):
        BATCH_SIZE = 2048

        while True:
            with self.reservoir_lock:
                datasets = self.reservoir.sample(BATCH_SIZE)

            ins = np.reshape(datasets[0], [BATCH_SIZE] + self.nn.input_shape)
            policies = np.reshape(datasets[1], [BATCH_SIZE, 69 * 5 * 5])
            values = np.reshape(datasets[2], [BATCH_SIZE, 1])

            self.training_data.put((ins, policies, values))

    def collect_records(self):
        print('Ready')
        log_file = open('connection_log.txt', 'w')

        nn = self.nn
        nn_lock = self.nn_lock
        checkpoint_weights = self.checkpoint_weights
        reservoir = self.reservoir
        reservoir_lock = self.reservoir_lock
        update_record_num = self.update_record_num
        new_record_count = self.new_record_count
        new_record_count_lock = self.new_record_count_lock

        class handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/weight':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

                    with nn_lock:
                        data = _pickle.dumps(checkpoint_weights, protocol=4)

                    self.wfile.write(data)

                    log_file.write('[{}] send the parameters\n'.format(
                        datetime.datetime.now(datetime.timezone.utc)))
                    log_file.flush()

                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

            def do_POST(self):
                if self.path == '/record':
                    content_length = int(self.headers.get('content-length'))
                    game_record = _pickle.loads(
                        self.rfile.read(content_length))
                    with reservoir_lock:
                        reservoir.push(simplejson.dumps(game_record.to_dict()))

                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

                    log_file.write('[{}] received a game record\n'.format(
                        datetime.datetime.now(datetime.timezone.utc)))
                    log_file.flush()

                    if update_record_num > 0:
                        with new_record_count_lock:
                            new_record_count[0] += 1

                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            pass

        with ThreadedHTTPServer(('', self.port), handler) as httpd:
            httpd.serve_forever()

    def update_parameters(self):
        for _ in range(2):
            sample_thread = threading.Thread(target=self._sample_datasets)
            sample_thread.start()

        log_file = open('training_log.txt', 'w')

        position = minishogilib.Position()
        position.set_start_position()
        init_position_nn_input = self.nn.get_inputs([position])

        while True:
            if self.update_record_num > 0:
                while True:
                    with self.new_record_count_lock:
                        if self.new_record_count[0] >= self.update_record_num:
                            self.new_record_count[0] -= self.update_record_num
                            break

            ins, policies, values = self.training_data.get()

            # Update neural network parameters.
            with self.nn_lock:
                if self.nn.iter() < 100000:
                    learning_rate = 1e-1
                elif self.nn.iter() < 200000:
                    learning_rate = 1e-2
                elif self.nn.iter() < 300000:
                    learning_rate = 1e-3
                else:
                    learning_rate = 1e-4

                loss = self.nn.step(
                    ins, policies, values, learning_rate)
                init_policy, init_value = self.nn.predict(
                    init_position_nn_input)

                if self.nn.iter() % 1000 == 0:
                    self.nn.save('./weights/iter_{}.h5'.format(self.nn.iter()))
                    self.checkpoint_weights = self.nn.get_weights()

            log_file.write('{}, {}, {}, {}, {}, {}\n'.format(datetime.datetime.now(
                datetime.timezone.utc), self.nn.iter(), loss['loss'], loss['policy_loss'], loss['value_loss'], init_value[0][0]))
            log_file.flush()

    def run(self):
        # Make the server which receives game records by selfplay from clients.
        collect_records_thread = threading.Thread(target=self.collect_records)
        collect_records_thread.start()

        # Update the neural network parameters.
        if not self.store_only:
            self.update_parameters()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--port', dest='port', type='int',
                      default=10055, help='port')
    parser.add_option('-s', '--store', action='store_true', dest='store', default=False,
                      help='Only store game records. Training will not be conducted.',)
    parser.add_option('-r', '--record_file', dest='record_file',
                      default='./records.json', help='Game records already played')
    parser.add_option('-w', '--weight_file', dest='weight_file',
                      default=None, help='Weights of neural network parameters')
    parser.add_option('-u', '--update_record_num', dest='update_record_num', type='int', default=0,
                      help='Update neural network parameters once every this number of records. If 0, it will update neural network parameters continuously.')

    (options, args) = parser.parse_args()

    trainer = Trainer(options.port, options.store,
                      options.record_file, options.weight_file, options.update_record_num)
    trainer.run()
