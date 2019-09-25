import tensorflow as tf

import network


def main():
    nn = network.Network()
    tf.keras.utils.plot_model(
        nn.model, show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    main()
