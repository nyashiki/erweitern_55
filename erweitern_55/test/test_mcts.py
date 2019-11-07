import minishogilib
import unittest

import mcts
import network


class TestMCTSMethods(unittest.TestCase):
    def test_nextmove1(self):
        nn = network.Network('gpu', 'DenseNet')
        nn.load('./weights.h5')

        print('iter:', nn.iter())

        config = mcts.Config()
        config.simulation_num = 12800

        search = mcts.MCTS(config)
        search.clear()

        position = minishogilib.Position()
        position.set_sfen('1r2k/2+r2/1g1Gp/PS3/KB3 b Bs 1')
        position.print()

        root = search.run(position, nn, 1e9, True)
        best_move = search.best_move(root)

        self.assertEqual(best_move.sfen(), 'B*2b')

if __name__ == '__main__':
    unittest.main()
