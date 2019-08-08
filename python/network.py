import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

import numpy as np


INPUT_CHANNEL = 134


def move_to_policy_index(color, move):
    """
    Convert a move (type: minishogilib.Move) into policy index.
    """

    if move.get_amount == 0:
        # In case of drawpping a prisoner.
        move_to = (move.get_to() // 5, move.get_to() %
                   5) if color == 0 else (4 - move.get_to() // 5, 4 - move.get_to() % 5)
        return (64 + move.get_hand_index(), move_to[0], move_to[1])
    else:
        # In case of moving a piece on the board.
        move_from = (move.get_from() // 5, move.get_from() %
                     5) if color == 0 else (4 - move.get_from() // 5, 4 - move.get_from() % 5)
        if move.get_promotion():
            # In case of promotion
            return (32 + 4 * move.get_direction() + (move.get_amount() - 1), move_from[0], move_from[1])
        else:
            return (4 * move.get_direction() + (move.get_amount() - 1), move_from[0], move_from[1])


class Network:
    def __init__(self):
        # Keras config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

        # Input layer
        input_image = keras.layers.Input(
            shape=[5, 5, INPUT_CHANNEL], dtype='float32')

        # Convolution layer
        x = keras.layers.Conv2D(
            256, [3, 3], padding='same', activation=tf.nn.relu)(input_image)

        # Residual blocks
        for i in range(11):
            x = self._residual_block(x)

        # Policy head
        policy = keras.layers.Conv2D(
            256, [3, 3], padding='same', activation=tf.nn.relu)(x)
        policy = keras.layers.Conv2D(
            69, [3, 3], padding='same', activation=tf.nn.relu)(policy)
        policy = keras.layers.Flatten()(policy)
        policy = keras.layers.Softmax(name='policy')(policy)

        # Value head
        value = keras.layers.Conv2D(
            1, [3, 3], padding='same', activation=tf.nn.relu)(x)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(256, activation=tf.nn.relu)(value)
        value = keras.layers.Dense(
            1, activation=tf.nn.tanh, name='value')(value)

        # define the model
        self.model = keras.Model(inputs=input_image, outputs=[policy, value])

        # optimizerを定義
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, clipnorm=1),
                           loss={'policy': keras.losses.categorical_crossentropy,
                                 'value': keras.losses.mean_squared_error},
                           loss_weights={'policy': 1, 'value': 1})

    def _residual_block(self, input_image, conv_kernel_shape=[3, 3]):
        conv_filters = int(input_image.shape[3])

        x = keras.layers.Conv2D(
            conv_filters, conv_kernel_shape, padding='same')(input_image)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(
            conv_filters, conv_kernel_shape, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, input_image])
        x = keras.layers.Activation('relu')(x)

        return x

    def step(self, train_images, policy_labels, value_labels, learning_rate=0.01):
        """Train the neural network one step.

        train_images: [B, C, H, W] order
        policy_labels: [B, C * H * W] order
        value_labels: [B, 1] order
        """

        # transpose [B, C, H, W] order to [B, H, W, C] order
        train_images = np.transpose(train_images, axes=[0, 2, 3, 1])
        policy_labels = np.reshape(policy_labels, (-1, 69, 5, 5))
        policy_labels = np.transpose(policy_labels, axes=[0, 2, 3, 1])
        policy_labels = np.reshape(policy_labels, (-1, 5 * 5 * 69))

        # set the learning rate
        K.set_value(self.model.optimizer.lr, learning_rate)

        loss = self.model.train_on_batch(
            train_images, [policy_labels, value_labels])
        return loss

    def predict(self, images):
        images = np.transpose(images, axes=[0, 2, 3, 1])
        policy, value = self.model.predict(
            images, batch_size=len(images), verbose=0, steps=None)

        # transpose [B, H, W, C] order to [B, C, H, W] order
        policy = np.reshape(policy, (-1, 5, 5, 69))
        policy = np.transpose(policy, axes=[0, 3, 1, 2])
        policy = np.reshape(policy, (-1, 69 * 5 * 5))

        return policy, value

    def load(self, filepath):
        self.model = keras.models.load_model(filepath, compile=True)
