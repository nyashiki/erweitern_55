import minishogilib
import numpy as np
import os
from optparse import OptionParser
import sys
import tensorflow as tf
import threading

import mcts
import network


class USI:
    def __init__(self, weight_file):
        self.weight_file = weight_file
        self.nn = None
        self.search = None

        self.option = {
            'ponder': False,
            'softmax_sampling_moves': 30
        }

    def isready(self):
        if self.nn is None:
            self.nn = network.Network()

            if self.weight_file is not None:
                self.nn.load(self.weight_file)

        self.config = mcts.Config()
        self.config.simulation_num = int(1e9)
        self.config.reuse_tree = True

        if self.search is None:
            self.search = mcts.MCTS(self.config)
        self.search.clear()

        self.position = minishogilib.Position()

        # ponder
        self.ponder_thread = None

    def start(self):
        while True:
            line = input()

            if not line:
                continue

            command = line.split()

            if command[0] == 'usi':
                print('id name erweitern_55')
                print('id author nyashiki')
                print('usiok')
            elif command[0] == 'setoption':
                key = command[2]
                value = command[4]

                if value == 'true' or value == 'True':
                    value = True
                elif value == 'false' or value == 'False':
                    value = False
                elif value.isdigit():
                    value = int(value)

                self.option[key.lower()] = value

            elif command[0] == 'position':
                self.ponder_stop()

                if command[1] == 'sfen':
                    sfen_kif = ' '.join(command[2:])
                    self.position.set_sfen(sfen_kif)

                elif command[1] == 'startpos':
                    self.position.set_start_position()

                else:
                    print('ERROR: Unknown protocol.')

            elif command[0] == 'isready':
                self.isready()
                print('readyok')

            elif command[0] == 'usinewgame':
                pass

            elif command[0] == 'go':
                timelimit = {}
                for (i, val) in enumerate(command):
                    if val == 'btime':
                        timelimit['btime'] = int(command[i + 1])
                    elif val == 'wtime':
                        timelimit['wtime'] = int(command[i + 1])
                    elif val == 'byoyomi':
                        timelimit['byoyomi'] = int(command[i + 1])

                moves = self.position.generate_moves()
                if len(moves) == 0:
                    print('bestmove resign')

                else:
                    checkmate, checkmate_move = self.position.solve_checkmate_dfs(
                        7)

                    if checkmate:
                        best_move = checkmate_move
                    else:
                        remain_time = timelimit['btime'] if self.position.get_side_to_move(
                        ) == 0 else timelimit['wtime']
                        think_time = remain_time // 20
                        if think_time < timelimit['byoyomi']:
                            think_time += timelimit['byoyomi'] + 700
                        think_time = max(think_time, 1700)

                        print('info string think time {}'.format(
                            think_time), flush=True)
                        root = self.search.run(
                            self.position, self.nn, think_time, True)

                        if self.position.get_ply() < self.option['softmax_sampling_moves']:
                            best_move = self.search.softmax_sample_among_top_moves(
                                root)
                        else:
                            best_move = self.search.best_move(root)

                    print('bestmove {}'.format(best_move), flush=True)

                    self.position.do_move(best_move)
                    self.ponder_start()

            elif command[0] == 'quit':
                os._exit(0)

            else:
                print('ERROR: Unknown command.', command[0])

    def ponder_start(self):
        """
            position: This position turn should be the other player's.
        """
        self.ponder_thread = threading.Thread(
            target=self.search.run, args=(self.position, self.nn, 0, True, not self.option['ponder']))
        self.ponder_thread.start()

    def ponder_stop(self):
        if self.ponder_thread is not None:
            self.search.stop()
            self.ponder_thread.join()
            self.ponder_thread = None


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-w', '--weight_file', dest='weight_file',
                      default=None, help='Weights of neural network parameters')
    (options, args) = parser.parse_args()

    usi = USI(options.weight_file)
    usi.start()
