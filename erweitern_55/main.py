import minishogilib

import math
import copy
import collections
from operator import itemgetter
import time

import numpy as np

import mcts
import network
import selfplay

import train


def main():
    neural_network = network.Network()
    neural_network.load('./weights/iter_85000.h5')

    # do predict once because the first prediction takes more time than latter one
    random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
    neural_network.predict(random_input)

    config = mcts.Config()
    config.simulation_num = 800
    # config.use_dirichlet = True

    search = mcts.MCTS(config)
    selfplay.run(neural_network, search, selfplay.SelfplayConfig(), True)


if __name__ == '__main__':
    print(minishogilib.__version__)

    # fix the seed
    np.random.seed(0)

    main()
