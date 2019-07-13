import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np

import minishogilib
import network
import sys

def move_to_policy_index(move):
    """
    Convert a move (type: minishogilib.Move) into policy index.
    """

    if move.get_amount == 0:
        # In case of drawpping a prisoner.
        return (64 + move.get_hand_index(), move.get_to() // 5, move.get_to() % 5)
    else:
        # In case of moving a piece on the board.
        if move.get_promotion():
            # In case of promotion
            return (32 + 4 * move.get_direction() + (move.get_amount() - 1), move.get_from() // 5, move.get_from() % 5)
        else:
            return (4 * move.get_direction() + (move.get_amount() - 1), move.get_from() // 5, move.get_from() % 5)

def load_teacher_onehot(filepath):
    """
    Load sfen kifs and return positions and moves.

    The file should be a csv file and be formatted as follows, and in which file a line represents a game.
        - timestamp, # ply, comment, sfen kif
    """

    MAX_ENTRY = 10000 # 500000 # 200000
    positions = np.zeros((MAX_ENTRY, 266, 5, 5))
    policy_labels = np.zeros((MAX_ENTRY, 69, 5, 5))
    value_labels = np.zeros((MAX_ENTRY, 1))

    with open(filepath) as f:
        line = f.readline()
        line_count = 0
        position_count = 0

        prev_nninput = np.zeros((266, 5, 5))

        while line:
            if position_count == MAX_ENTRY:
                break

            position = minishogilib.Position()
            position.set_start_position()

            sfen = line.split(',')[3]
            sfen_split = sfen.split()

            ply = 0
            win_color = 0 if len(sfen_split) % 2 == 1 else 1

            for sfen_move in sfen_split:
                if position_count == MAX_ENTRY:
                    break

                move = position.sfen_to_move(sfen_move)
                index = move_to_policy_index(move)

                onehot_policy = np.zeros((69, 5, 5))
                onehot_policy[index] = 1

                nn_input = np.array(position.to_nninput(1))
                nn_input = np.concatenate([nn_input, prev_nninput[2:233]])

                positions[position_count] = nn_input
                policy_labels[position_count] = onehot_policy
                value_labels[position_count] = 1 if (ply % 2) == win_color else -1

                position.do_move(move)

                prev_nninput = nn_input
                position_count += 1
                ply += 1

            line_count += 1
            sys.stdout.write('\rload ' + str(line_count) + ' lines (' + str(len(positions)) + ' positions).')

            line = f.readline()

        sys.stdout.write(' ... done.\n')

    return positions, policy_labels, value_labels

def main():
    # Construct a neural network
    neural_network = network.Network()

    # Output a png file of the network
    # plot_model(neural_network.model, show_shapes=True, show_layer_names=True, to_file='model.png')

    # Output information about the network
    neural_network.model.summary()

    positions, policy_labels, value_labels = load_teacher_onehot('./kif.txt')

    # tranpose it into tensorflow-like format
    positions = np.transpose(positions, axes=[0, 2, 3, 1])
    policy_labels = np.transpose(policy_labels, axes=[0, 2, 3, 1])

    neural_network.train(positions, policy_labels, value_labels, 1024, 1)

if __name__ == '__main__':
    print(tf.__version__)
    tf.compat.v1.set_random_seed(1)

    main()
