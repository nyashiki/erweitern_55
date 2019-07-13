import tensorflow as tf
import network
from tensorflow.python.keras.utils.vis_utils import plot_model

import minishogilib

def move_to_policy_index(move):
    """
        Convert move (type: minishogilib.Move) into policy index.
    """

    if move.get_amount == 0:
        # In case of drawpping a prisoner.
        return (move.get_to() // 5, move.get_to() % 5, 64 + move.get_hand_index())
    else:
        # In case of moving a piece on the board.
        if move.get_promotion():
            # In case of promotion
            return (move.get_from() // 5, move.get_from() % 5, 32 + 4 * move.get_direction() + (move.get_amount() - 1))
        else:
            return (move.get_from() // 5, move.get_from() % 5, 4 * move.get_direction() + (move.get_amount() - 1))

def load_teacher_onehot(filepath):
    """
    Load sfen kifs and return positions and moves.

    The file should be a csv file and be formatted as follows, and in which file a line represents a game.
        - timestamp, # ply, comment, sfen kif
    """

    positions = np.zeros()
    labels = np.zeros()

    with open(filepath) as f:
        line = f.readline()

        while line:
            position = minishogilib.Position()
            position.set_start_position()

            sfen = line.split(',')[3]
            sfen_split = sfen.split()

            for sfen_move in sfen_split:
                move = position.sfen_to_move(sfen_move)
                index = move_to_policy_index(move)

                onehot_policy = np.zeros((5, 5, 69))
                onehot_policy[index] = 1

                np.append(positions, position.to_nninput(), axis=0)
                np.append(labels, onehot_policy, axis=0)

                position.do_move(move)

            line = f.readline()

def main():
    # Construct a neural network
    neural_network = network.Network()

    # Output a png file of the network
    # plot_model(neural_network.model, show_shapes=True, show_layer_names=True, to_file='model.png')

    # Output information about the network
    neural_network.model.summary()

if __name__ == '__main__':
    print(tf.__version__)
    tf.compat.v1.set_random_seed(1)

    main()
