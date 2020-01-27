from optparse import OptionParser
import os
import _pickle
import sys
import threading
import urllib.request

import mcts
import network
import selfplay


class Client:
    """Client that connects the server and conducts selfplay games.
    """

    def __init__(self, ip, port, update=True, cpu_only=False, update_iter=1, random_play=False, checkmate_depth=7):
        self.host = ip
        self.port = port
        self.nn = None
        self.update = update
        self.cpu_only = cpu_only
        self.update_iter = update_iter
        self.random_play = random_play
        self.checkmate_depth = checkmate_depth

    def run(self):
        mcts_config = mcts.Config()
        mcts_config.batch_size = 16
        mcts_config.simulation_num = 800
        mcts_config.forced_playouts = False
        mcts_config.use_dirichlet = True
        mcts_config.reuse_tree = True
        mcts_config.target_pruning = False
        mcts_config.immediate = False

        search = mcts.MCTS(mcts_config)

        device = 'cpu' if self.cpu_only else 'gpu'
        self.nn = network.Network(device)

        iter = 0

        while True:
            # Ask the server the current neural network parameters.
            if self.update and iter % self.update_iter == 0:
                url = 'http://{}:{}/weight'.format(self.host, self.port)
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as res:
                    weights = _pickle.loads(res.read())
                    self.nn.model.set_weights(weights)

            # Conduct selfplay.
            if self.random_play:
                game_record = selfplay.random_play(
                    stop_with_checkmate=False, trim_checkmate=False)

            else:
                search.clear()
                game_record = selfplay.run(
                    self.nn, search, checkmate_depth=self.checkmate_depth, stop_with_checkmate=False, trim_checkmate=False)

            # Send result.
            url = 'http://{}:{}/record'.format(self.host, self.port)
            data = _pickle.dumps(game_record, protocol=4)
            req = urllib.request.Request(url, data)
            with urllib.request.urlopen(req) as res:
                pass

            iter += 1


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
    parser.add_option('-u', '--update-iter', dest='update_iter', type='int',
                      default=1, help='The iteration to update neural network parameters.')
    parser.add_option('-r', '--random-play', action='store_true', dest='random_play',
                      default=False, help='Games are conducted by random play.')
    parser.add_option('-d', '--checkmate-depth', dest='checkmate_depth',
                      default=7, help='The depth of checkmate searching.')

    (options, args) = parser.parse_args()

    client = Client(options.ip, options.port,
                    not options.no_update, options.cpu_only, options.update_iter, options.random_play, options.checkmate_depth)
    client.run()
