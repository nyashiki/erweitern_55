from unittest import TestCase
from unittest import TestLoader
import numpy as np

from erweitern_55 import network

class TestNetwork(TestCase):
    def test_initialization(self):
        nn = network.Network()

    def test_zero_input(self):
        nn = network.Network()
        ins = np.zeros([128, 266, 5, 5])
        nn.predict(ins)
