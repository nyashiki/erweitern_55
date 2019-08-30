import minishogilib
import numpy as np

import mcts
import network

def main():
    neural_network = network.Network()
    neural_network.load('./weights/iter_30000.h5')

    # do predict once because the first prediction takes more time than latter one
    random_input = np.random.rand(1, network.INPUT_CHANNEL, 5, 5)
    neural_network.predict(random_input)

    config = mcts.Config()
    config.simulation_num = 800
    config.reuse_tree = False

    search = mcts.MCTS(config)
    search.clear()

    position = minishogilib.Position()

    while True:
        line = input()

        if not line:
          continue

        command = line.split()

        if command[0] == 'usi':
            print('id name erweitern_55')
            print('id author nyashiki')
            print('usiok')

        elif command[0] == 'position':
            if command[1] == 'sfen':
                sfen_kif = ' '.join(command[2:])
                position.set_sfen(sfen_kif)

            else:
                print('ERROR: Unknown protocol.')

        elif command[0] == 'isready':
            print('readyok')

        elif command[0] == 'usinewgame':
            pass

        elif command[0] == 'go':
            # ToDo: timelimit

            root = search.run(position, neural_network)
            print('bestmove {}'.format(search.best_move(root)))

        else:
            print('ERROR: Unknown command.')

if __name__ == '__main__':
    # fix the seed
    np.random.seed(0)

    main()
