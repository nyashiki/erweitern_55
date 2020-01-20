from unittest import TestCase

import minishogilib


class TestCheckmate(TestCase):
    def test_checkmate1(self):
        position = minishogilib.Position()
        position.set_sfen('2k2/5/2P2/5/2K2 b G 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, True)

    def test_checkmate2(self):
        position = minishogilib.Position()
        position.set_sfen('5/5/2k2/5/2K2 b 2GS 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, True)

    def test_checkmate3(self):
        position = minishogilib.Position()
        position.set_sfen('5/5/2k2/5/2K2 b 2G 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, False)

    def test_checkmate4(self):
        position = minishogilib.Position()
        position.set_sfen('2k2/5/2B2/5/2K2 b GSBRgsr2p 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, True)

    def test_checkmate5(self):
        position = minishogilib.Position()
        position.set_sfen('2G1k/5/4G/5/2K2 b P 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, False)

    def test_checkmate6(self):
        position = minishogilib.Position()
        position.set_sfen('4k/5/4B/5/2K1R b - 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, True)

    def test_checkmate7(self):
        position = minishogilib.Position()
        position.set_sfen('4k/4p/5/5/K4 b BG 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, True)

    def test_checkmate8(self):
        position = minishogilib.Position()
        position.set_sfen('5/4k/3pp/5/K4 b RG 1')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, True)

    def test_checkmate9(self):
        position = minishogilib.Position()
        position.set_sfen('rbsgk/4p/5/P4/KGSBR b - 1 moves 2e3d 2a2b 4e4d 4a3b 3d4e 5a4a 1e2e 2b2a 5d5c 3a2b 3e3d 3b1d 2e2d 1d3b 4d3c 2b3c 3d3c G*2b 3c2b 2a2b S*4d S*2a G*3e 3b4c 2d2b 2a2b 4d4c 4a4c 4e5d S*3b B*4d R*3a G*4e 4c4d 3e4d 3b2c R*2e B*3d 2e3e 3d4e+ 4d4e 3a3e+ 4e3e R*3a R*4e 2b3c B*1e G*3b 1e3c 3b3c S*4b 3c3b 4b3a 3b3a 3e4d S*3d R*5a 3d4e+ 5d4e R*3e S*3c B*4a 5c5b 3e2e+ 5e5d 2c3b 3c3b 4a3b S*4c 3b4a 4d3d S*4b 4c4b 3a4b 5a4a 4b4a B*4d 1a2a 3d3c R*2d S*2b 2d2b 3c2b 2e2b 4d2b S*4c 5d5e S*3d R*2d 3d4e 2b1a+ 2a3b 1a2a 3b4b 2a4c 4b4c S*4d 4c5b R*5c 5b4b 5e4e 4b3a S*2b 3a4b 2b3c 4b3a 3c2b 3a4b 2b3c 4b3a 3c2b 3a4b 5c4c 4b5b')

        checkmate, _ = position.solve_checkmate_dfs(7)

        self.assertEqual(checkmate, False)
