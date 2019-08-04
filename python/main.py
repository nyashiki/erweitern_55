import minishogilib

import math
import copy
import collections
from operator import itemgetter
import time

import numpy as np

import mcts
from nn import network

def main():
    position = minishogilib.Position()
    position.set_start_position()

    neural_network = network.Network()
    neural_network.load('./nn/weights/epoch_030.h5')

    # do predict once because the first prediction takes more time than latter one
    random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
    neural_network.predict(random_input)

    config = mcts.Config()
    search = mcts.MCTS(config)

    while True:
        start_time = time.time()

        checkmate, checkmate_move = position.solve_checkmate_dfs(7)
        if checkmate:
            best_move = checkmate_move
        else:
            root = search.run(position, neural_network)
            best_move = search.best_move(root)
            search.dump(root)

        elapsed = time.time() - start_time

        if best_move.is_null_move():
            break

        position.do_move(best_move)

        print('--------------------')
        position.print()
        print(best_move)
        print('time:', elapsed)
        print('--------------------')

if __name__ == '__main__':
    print(minishogilib.__version__)

    # fix the seed
    np.random.seed(0)

    main()
