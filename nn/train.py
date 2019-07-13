import tensorflow as tf
import network
from tensorflow.python.keras.utils.vis_utils import plot_model

def main():
    # ニューラルネットワークを構築
    neural_network = network.Network()

    # モデルをpngに書き出す
    # plot_model(neural_network.model, show_shapes=True, show_layer_names=True, to_file='model.png')

    # パラメータ数などの情報を表示
    neural_network.model.summary()

if __name__ == '__main__':
    print(tf.__version__)
    tf.compat.v1.set_random_seed(1)

    main()
