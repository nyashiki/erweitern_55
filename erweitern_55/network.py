import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

from multiprocessing import Pool
import numpy as np
import os
import psutil

REGULARIZER_c = 1e-4

class Network:
    def __init__(self, cpu=False):
        # Keras config
        if cpu:
            # CPU settings.
            cpu_count = psutil.cpu_count(logical=False)
            config = tf.ConfigProto(device_count={'CPU': cpu_count})
            config.intra_op_parallelism_threads = cpu_count
            config.inter_op_parallelism_threads = 1
            config.allow_soft_placement = True
            os.environ['KMP_BLOCKTIME'] = '1'
            os.environ['KMP_HW_SUBSET'] = '1t'
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)

        else:
            # GPU settings.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

        self.network_type = None
        self.input_shape = None

        # Construct the network.
        # ins, policy, value = self._alphazero_network()
        # ins, policy, value = self._kp_network()
        ins, policy, value = self._dense_network()

        # Define the model.
        self.model = keras.Model(inputs=ins, outputs=[policy, value])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1, momentum=0.9),
                           loss={'policy': keras.losses.CategoricalCrossentropy(from_logits=True),
                                 'value': keras.losses.mean_squared_error})

        # For multithreading.
        self.model._make_predict_function()
        self.session = tf.compat.v1.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()

        # Do predict once because the first prediction takes more time than latter one.
        random_input = np.random.rand(*([1] + self.input_shape))
        self.predict(random_input)

    def _alphazero_network(self):
        """ Construct AlphaZero-like network.

        # Returns:
            input_image: the input layer of this network.
            policy: the policy head layer.
            value : the value head layer.
        """

        self.network_type = 'AlphaZero'
        self.input_shape = [266, 5, 5]

        # Input layer.
        input_image = keras.layers.Input(
            shape=self.input_shape, dtype='float32')

        # Convolution layer.
        x = keras.layers.Conv2D(
            128, [3, 3], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(input_image)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)

        # Residual blocks.
        for _ in range(11):
            x = self._residual_block(x)

        # Policy head.
        policy = keras.layers.Conv2D(
            128, [3, 3], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        policy = keras.layers.BatchNormalization(axis=1)(policy)
        policy = keras.layers.ReLU()(policy)
        policy = keras.layers.Conv2D(
            69, [1, 1], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(policy)
        policy = keras.layers.Flatten(name='policy')(policy)

        # Value head.
        value = keras.layers.Conv2D(
            1, [1, 1], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        value = keras.layers.BatchNormalization(axis=1)(value)
        value = keras.layers.ReLU()(value)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(
            REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(value)
        value = keras.layers.Dense(
            1, activation=tf.nn.tanh, name='value', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(value)

        return input_image, policy, value

    def _kp_network(self):
        """Construct King-Piece relationship network.

        # Returns:
            input_image: the input layer of this network.
            policy: the policy head layer.
            value : the value head layer.
        """

        self.network_type = 'KP'
        self.input_shape = [23763]

        # Input layer.
        input_image = keras.layers.Input(
            shape=self.input_shape, dtype='float32')

        x = input_image
        for _ in range(3):
            x = keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(
                REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(x)
            x = keras.layers.BatchNormalization()(x)

        # Policy head.
        policy = keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(
            REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(x)
        policy = keras.layers.BatchNormalization()(policy)
        policy = keras.layers.Dense(69 * 5 * 5, activation=None, kernel_regularizer=regularizers.l2(
            REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), name='policy')(policy)

        # Value head.
        value = keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(
            REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(x)
        value = keras.layers.BatchNormalization()(value)
        value = keras.layers.Dense(
            1, activation=tf.nn.tanh, name='value', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(value)

        return input_image, policy, value

    def _dense_network(self):
        self.network_type = 'DenseNet'
        self.input_shape = [266, 5, 5]

        # Input layer
        input_image = keras.layers.Input(
            shape=self.input_shape, dtype='float32')

        # Convolution layer
        x = keras.layers.Conv2D(
            128, [3, 3], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(input_image)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)

        # Dense blocks.
        for _ in range(11):
            x = self._dense_block(x)

        # Policy head.
        policy = keras.layers.Conv2D(
            128, [3, 3], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        policy = keras.layers.BatchNormalization(axis=1)(policy)
        policy = keras.layers.ReLU()(policy)

        policy = keras.layers.Conv2D(
            69, [1, 1], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(policy)
        policy = keras.layers.Flatten(name='policy')(policy)

        # Value head.
        value = keras.layers.Conv2D(
            1, [1, 1], padding='same', activation=None, kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        value = keras.layers.BatchNormalization(axis=1)(value)
        value = keras.layers.ReLU()(value)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(
            REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(value)
        value = keras.layers.Dense(
            1, activation=tf.nn.tanh, name='value', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(value)

        return input_image, policy, value

    def _residual_block(self, input_image, conv_kernel_shape=[3, 3]):
        """Construct a residual block.

        # Arguments:
            input_image: the input layer for this residual block.
            conv_kernel_shape: The shape of convolutional newtorks used in this residual block.

        # Returns:
            x: the output layer of this residual block.
        """
        conv_filters = int(input_image.shape[1])

        x = keras.layers.Conv2D(
            conv_filters, conv_kernel_shape, activation=None, padding='same', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(input_image)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(
            conv_filters, conv_kernel_shape, activation=None, padding='same', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.Add()([x, input_image])
        x = keras.layers.ReLU()(x)

        return x

    def _dense_block(self, input_image, k=12):
        """Construct a dense block.
        """

        x = keras.layers.Conv2D(128, [1, 1], activation=None, padding='same', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(input_image)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(k, [3, 3], activation=None, padding='same', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Concatenate(axis=1)([input_image, x])

        return x

    def step(self, train_images, policy_labels, value_labels, learning_rate=0.01):
        """Train the neural network one step.

        # Arguments:
            train_images: inputs used for traininig the neural network.
            policy_labels: teachers of the policy head.
            value_labels: teachers of the value head.

        # Returns:
            Dictionary composed of losses and metrics.
        """

        with self.session.as_default():
            with self.graph.as_default():
                # Set the learning rate.
                K.set_value(self.model.optimizer.lr, learning_rate)

                loss = self.model.train_on_batch(
                    x=train_images,
                    y={'policy': policy_labels,
                       'value': value_labels})

        return dict(zip(self.model.metrics_names, loss))

    def predict(self, images):
        """Get policy head and value head values.

        # Arguments:
            images: the neural network inputs.

        # Returns:
            policy: the value of policy head.
            value: the value of value head.
        """
        with self.session.as_default():
            with self.graph.as_default():
                policy, value = self.model.predict(
                    images, batch_size=len(images), verbose=0, steps=None)

        return policy, value

    def load(self, filepath):
        """Set the neural network weights and the optimizer from a file.

        # Arguments:
            filepath: the path of the saved weights file.
        """
        with self.session.as_default():
            with self.graph.as_default():
                self.model = keras.models.load_model(filepath, compile=True)

    def get_weights(self):
        """Get weights of the neural networks.

        # Returns:
            Weights of the neural networks.
        """
        with self.session.as_default():
            with self.graph.as_default():
                return self.model.get_weights()

    def set_weights(self, weights):
        """ Set weights of the neural networks.
        """
        with self.session.as_default():
            with self.graph.as_default():
                self.model.set_weights(weights)

    def save(self, filepath):
        """Save weights and the optimizer to a file.

        # Arguments:
            filepath: a file to which save the neural network weights and the optimizer.
        """
        with self.session.as_default():
            with self.graph.as_default():
                self.model.save(filepath, include_optimizer=True)

    def get_input(self, position, dim_4=False):
        shape = [1] + self.input_shape if dim_4 else self.input_shape

        if self.network_type == 'AlphaZero':
            return np.reshape(position.to_alphazero_input(), shape)
        elif self.network_type == 'KP':
            return np.reshape(position.to_kp_input(), shape)
        elif self.network_type == 'DenseNet':
            return np.reshape(position.to_alphazero_input(), shape)

    def get_inputs(self, positions):
        """Get neural network inputs' representation of the positions.

        # Arguments:
            positions: list of positions.

        # Returns:
            ins: neural network inputs' representation.
        """
        ins = np.zeros([len(positions)] + self.input_shape, dtype='float32')

        for i, position in enumerate(positions):
            ins[i] = self.get_input(position)

        return ins

    def iter(self):
        """Get the iteration of training.
        """
        with self.session.as_default():
            with self.graph.as_default():
                return K.get_value(self.model.optimizer.iterations)

    def zero_inputs(self, batch_size):
        """Get zero inputs.
        """
        return np.zeros([batch_size] + self.input_shape, dtype='float32')
