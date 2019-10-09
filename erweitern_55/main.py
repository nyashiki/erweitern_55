import minishogilib

import math
import copy
import collections
import numpy as np
from operator import itemgetter
import time

import mcts
import network
import selfplay
import train


def main():
    neural_network = network.Network()

    # do predict once because the first prediction takes more time than latter one
    random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
    neural_network.predict(random_input)

    config = mcts.Config()
    config.simulation_num = 800

    search = mcts.MCTS(config)
    selfplay.run(neural_network, search, True)


if __name__ == '__main__':
    print(minishogilib.__version__)

    # fix the seed
    np.random.seed(0)

    main()
