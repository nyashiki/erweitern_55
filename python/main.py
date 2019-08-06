import minishogilib

import math
import copy
import collections
from operator import itemgetter
import time

import numpy as np

import mcts
from nn import network
import selfplay

import train

def main():
    neural_network = network.Network()
    neural_network.load('./weights/iter_230000.h5')

    # do predict once because the first prediction takes more time than latter one
    random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
    neural_network.predict(random_input)

    search = mcts.MCTS(mcts.Config())
    selfplay.run(neural_network, search, selfplay.SelfplayConfig(), True)

if __name__ == '__main__':
    print(minishogilib.__version__)

    # fix the seed
    np.random.seed(0)

    main()
