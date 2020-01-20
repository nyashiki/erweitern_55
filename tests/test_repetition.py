from unittest import TestCase

import minishogilib


class TestRepetition(TestCase):
    def test_repetition1(self):
        position = minishogilib.Position()
        position.set_sfen('rbsgk/4p/5/P4/KGSBR b - 1')

        self.assertEqual(position.is_repetition(), (False, False, False))

    def test_repetition2(self):
        position = minishogilib.Position()
        position.set_sfen(
            'rbsgk/4p/5/P4/KGSBR b - 1 moves 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a')

        self.assertEqual(position.is_repetition(), (True, False, False))

    def test_repetition3(self):
        position = minishogilib.Position()
        position.set_sfen(
            'rbsgk/4p/5/P4/KGSBR b - 1 moves 3e2d 3a4b 2e3d 2a2b 4e4d 4a3b 5e4e 5a4a 3d5b 4a5a 5b3d 5a4a 3d5b 4a5a 5b2e 5a4a 2e5b 4a5a 5b3d 5a4a 3d5b')

        self.assertEqual(position.is_repetition(), (True, False, False))

    def test_repetition4(self):
        position = minishogilib.Position()
        position.set_sfen(
            '2k2/5/5/5/2K2 b R 1 moves R*3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c')

        self.assertEqual(position.is_repetition(), (True, False, True))

    def test_repetition5(self):
        position = minishogilib.Position()
        position.set_sfen('rbsgk/4p/5/P4/KGSBR b - 1 moves 4e4d 4a3b 2e3d 3a2b 3e2d 5a4a 5d5c 4a4b 5c5b 4b4d 5e4d G*1d 1e1d 3b1d R*1e 1d3b G*4b R*5d 4d4e 5d3d 4e3d B*3a 4b3b 2a3b 1e1b 1a1b R*1e 1b2a B*1b 2a1a 1b2c 1a2a 2c1b 2a1a 1b2c 1a2a 2c1b 2a1a 1b2c 1a2a 2c1b')

        self.assertEqual(position.is_repetition(), (True, False, True))

    def test_repetition6(self):
        position = minishogilib.Position()
        position.set_sfen(
            '3k1/5/2R2/5/2K2 b - 1 moves 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a')

        self.assertEqual(position.is_repetition(), (True, True, False))

    def test_repetition7(self):
        position = minishogilib.Position()
        position.set_sfen(
            'rbsgk/4p/5/P4/KGSBR b - 1 moves 5e4d 1a2b 4d5e 2b1a 5e4d 1a2b 4d5e 2b1a')

        self.assertEqual(position.is_repetition(), (False, False, False))

    def test_repetition8(self):
        position = minishogilib.Position()
        position.set_sfen(
            '2k2/5/5/5/2K2 b R 1 moves R*3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a 2c3c 3a2a 3c2c 2a3a')

        self.assertEqual(position.is_repetition(), (False, False, False))
