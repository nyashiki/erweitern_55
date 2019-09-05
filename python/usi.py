import minishogilib
import numpy as np
import os
import tensorflow as tf
import threading

import mcts
import network

class USI:
    def __init__(self):
        self.nn = network.Network()
        self.nn.load('./weights/iter_80000.h5')

        self.config = mcts.Config()
        self.config.simulation_num = 6400
        self.config.reuse_tree = True

        self.search = mcts.MCTS(self.config)
        self.search.clear()

        self.position = None

        # ponder
        self.ponder_thread = None

        self.ponder_config = mcts.Config()
        self.ponder_config.simulation_num = int(1e9)

    def start(self):
        # do predict once because the first prediction takes more time than latter one
        random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
        self.nn.predict(random_input)

        self.position = minishogilib.Position()

        while True:
            line = input()

            if not line:
                continue

            command = line.split()

            if command[0] == 'usi':
                print('id name erweitern_55(3-days)')
                print('id author nyashiki')
                print('usiok')

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
                print('readyok')

            elif command[0] == 'usinewgame':
                pass

            elif command[0] == 'go':
                # ToDo: timelimit

                moves = self.position.generate_moves()
                if len(moves) == 0:
                    print('bestmove resign')
                else:
                    root = self.search.run(self.position, self.nn)
                    best_move = self.search.best_move(root)
                    print('bestmove {}'.format(best_move))

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
        self.search.config = self.ponder_config
        self.ponder_thread = threading.Thread(target=self.search.run, args=(self.position, self.nn))
        self.ponder_thread.start()

    def ponder_stop(self):
        if self.ponder_thread is not None:
            self.search.stop()
            self.ponder_thread.join()
            self.ponder_thread = None

        self.search.config = self.config

if __name__ == '__main__':
    # fix the seed
    np.random.seed(0)

    usi = USI()
    usi.start()
