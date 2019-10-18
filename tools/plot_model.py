import tensorflow as tf

from erweitern_55 import network


def main():
    nn = network.Network()
    tf.keras.utils.plot_model(
        nn.model, show_shapes=True, show_layer_names=True)
    nn.model.summary()

if __name__ == '__main__':
    main()
