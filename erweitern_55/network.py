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
    def __init__(self, device='gpu'):
        tf.compat.v1.disable_eager_execution()

        # Keras config
        if device == 'cpu':
            # CPU settings.
            cpu_count = psutil.cpu_count(logical=False)
            config = tf.ConfigProto(device_count={'CPU': cpu_count})
            config.intra_op_parallelism_threads = cpu_count
            config.inter_op_parallelism_threads = 1
            config.allow_soft_placement = True
            os.environ['KMP_BLOCKTIME'] = '1'
            os.environ['KMP_HW_SUBSET'] = '1t'
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            sess = tf.Session(config=config)

        elif device == 'gpu':
            # GPU settings.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

        elif device == 'tpu':
            # Work in progress.
            tpu_address = "grpc://" + os.environ['COLAB_TPU_ADDR']
            sess = tf.Session(tpu_address)

        keras.backend.set_session(sess)

        self.network_type = None
        self.input_shape = None

        # Construct the network.
        ins, policy, value = self._alphazero_network()

        # Define the model.
        self.model = keras.Model(inputs=ins, outputs=[policy, value])

        if device == 'tpu':
            strategy = tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address))
            self.model = tf.contrib.tpu.keras_to_tpu_model(
                self.model, strategy=strategy)

        self.model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                           loss={'policy': keras.losses.CategoricalCrossentropy(from_logits=True),
                                 'value': keras.losses.mean_squared_error},
                           loss_weights={'policy': 1, 'value': 1})

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
            256, [3, 3], strides=1, padding='same', activation='linear', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(input_image)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)

        # Residual blocks.
        for _ in range(5):
            x = self._residual_block(x)

        # Policy head.
        policy = keras.layers.Conv2D(
            256, [3, 3], strides=1, padding='same', activation='linear', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        policy = keras.layers.BatchNormalization(axis=1)(policy)
        policy = keras.layers.ReLU()(policy)
        policy = keras.layers.Conv2D(
            69, [1, 1], strides=1, padding='same', activation='linear', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(policy)
        policy = keras.layers.Flatten(name='policy')(policy)

        # Value head.
        value = keras.layers.Conv2D(
            1, [1, 1], strides=1, padding='same', activation='linear', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        value = keras.layers.BatchNormalization(axis=1)(value)
        value = keras.layers.ReLU()(value)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(
            256, activation='linear', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c))(value)
        value = keras.layers.ReLU()(value)
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
            conv_filters, conv_kernel_shape, strides=1, activation='linear', padding='same', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(input_image)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(
            conv_filters, conv_kernel_shape, strides=1, activation='linear', padding='same', kernel_regularizer=regularizers.l2(REGULARIZER_c), bias_regularizer=regularizers.l2(REGULARIZER_c), data_format='channels_first')(x)
        x = keras.layers.BatchNormalization(axis=1)(x)
        x = keras.layers.Add()([x, input_image])
        x = keras.layers.ReLU()(x)

        return x

    def step(self, train_images, policy_labels, value_labels, assertion=False):
        """Train the neural network one step.

        # Arguments:
            train_images: inputs used for traininig the neural network.
            policy_labels: teachers of the policy head.
            value_labels: teachers of the value head.

        # Returns:
            Dictionary composed of losses and metrics.
        """

        if assertion:
            assert not np.isinf(train_images).any(
            ), 'Inf is detected in train_images.'
            assert not np.isinf(policy_labels).any(
            ), 'Inf is detected in policy_labels.'
            assert not np.isnan(train_images).any(
            ), 'NaN is detected in train_images.'
            assert not np.isnan(policy_labels).any(
            ), 'NaN is detected in policy_labels.'
            assert ((train_images >= 0.0) & (train_images <= 1.0)).all(
            ), 'There is a value out of [0, 1] in train_images.'
            assert ((policy_labels >= 0.0) & (policy_labels <= 1.0)).all(
            ), 'There is a value out of [0, 1] in policy_labels.'

            for policy_label in policy_labels:
                assert abs(np.sum(
                    policy_label) - 1.0) < 1e-4, 'np.sum(policy_label) != 1 ({}).'.format(np.sum(policy_label))

        with self.session.as_default():
            with self.graph.as_default():
                # Set the learning rate.
                if self.iter() < 50000:
                    learning_rate = 2e-1
                elif self.iter() < 100000:
                    learning_rate = 2e-2
                elif self.iter() < 150000:
                    learning_rate = 2e-3
                else:
                    learning_rate = 2e-4

                K.set_value(self.model.optimizer.lr, learning_rate)

                loss = self.model.train_on_batch(
                    x=train_images,
                    y={'policy': policy_labels,
                       'value': value_labels})

                result = dict(zip(self.model.metrics_names, loss))

        return result

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
