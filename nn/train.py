import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy as np

import minishogilib
import network
import random
import sys

def load_teacher_onehot(filepath):
    """
    Load sfen kifs and return positions and moves.

    The file should be a csv file and be formatted as follows, and in which file a line represents a game.
        - timestamp, # ply, comment, sfen kif
    """

    MAX_ENTRY = 3000000
    inputs = np.zeros((MAX_ENTRY, network.INPUT_CHANNEL, 5, 5), dtype='float32')
    policy = np.zeros((MAX_ENTRY, 69, 5, 5), dtype='float32')
    value = np.zeros((MAX_ENTRY, 1), dtype='float32')

    with open(filepath) as f:
        line = f.readline()
        line_count = 0
        position_count = 0

        while line:
            if position_count == MAX_ENTRY:
                break

            position = minishogilib.Position()
            position.set_start_position()

            sfen = line.split(',')[3]
            sfen_split = sfen.split()

            ply = 0
            win_color = 0 if len(sfen_split) % 2 == 1 else 1

            target_ply = random.randrange(len(sfen_split))

            for sfen_move in sfen_split:
                if position_count == MAX_ENTRY:
                    break

                move = position.sfen_to_move(sfen_move)

                index = network.move_to_policy_index(position.get_side_to_move(), move)

                onehot_policy = np.zeros((69, 5, 5))
                onehot_policy[index] = 1

                nn_input = np.array(position.to_nninput()).reshape(network.INPUT_CHANNEL, 5, 5)
                inputs[position_count] = nn_input
                policy[position_count] = onehot_policy
                value[position_count] = 1 if (ply % 2) == win_color else -1
                position_count += 1

                position.do_move(move)

                ply += 1

            line_count += 1
            sys.stdout.write('\rload ' + str(line_count) + ' lines (' + str(position_count) + ' positions).')

            line = f.readline()

        sys.stdout.write(' ... done.\n')

    return inputs[:position_count], policy[:position_count], value[:position_count]

def main():
    # Construct a neural network
    neural_network = network.Network()

    # Output a png file of the network
    plot_model(neural_network.model, show_shapes=True, show_layer_names=True, to_file='model.png')

    # Output information about the network
    neural_network.model.summary()

    inputs, policy, value = load_teacher_onehot('./kif.txt')

    # tranpose it into tensorflow-like format (i.e. BHWC order)
    inputs = np.transpose(inputs, axes=[0, 2, 3, 1])
    policy = np.transpose(policy, axes=[0, 2, 3, 1])
    policy = np.reshape(policy, (-1, 5 * 5 * 69))

    batch_size = 1024
    batch_num_per_epoch = len(inputs) // batch_size

    for epoch in range(1000):
        random_indices = list(range(len(inputs)))
        random.shuffle(random_indices)

        for b in range(batch_num_per_epoch):
            start_index = b * batch_size
            end_index = start_index + batch_size

            inputs_ = inputs[random_indices[start_index:end_index]]
            policy_ = policy[random_indices[start_index:end_index]]
            value_ = value[random_indices[start_index:end_index]]

            loss = neural_network.step(inputs_, policy_, value_)
            print(loss)

        # save the model
        filename = './weights/epoch_{:03}.h5'.format(epoch)
        neural_network.model.save(filename, include_optimizer=True)

if __name__ == '__main__':
    print(tf.__version__)
    print(minishogilib.version())

    tf.compat.v1.set_random_seed(1)

    main()
