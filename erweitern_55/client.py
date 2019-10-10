import socketio
from optparse import OptionParser
import os
import _pickle
import sys

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

        url = 'http://{}:{}'.format(self.host, self.port)
        sio = socketio.Client()

        @sio.event
        def connect():
            sio.emit('parameter')

        @sio.event
        def disconnect():
            os._exit(0)

        @sio.on('receive_parameter')
        def receive_parameter(data):
            if self.nn is None or self.update:
                if self.nn is None:
                    self.nn = network.Network(self.cpu_only)

                weights = _pickle.loads(data)
                self.nn.set_weights(weights)

            # Conduct selfplay.
            search.clear()
            game_record = selfplay.run(self.nn, search)

            # Send game result.
            data = _pickle.dumps(game_record, protocol=4)
            sio.emit('record', data)

            # Ask current parameters again.
            if self.update:
                sio.emit('parameter')
            else:
                receive_parameter(None)

        sio.connect(url)
        sio.wait()

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
