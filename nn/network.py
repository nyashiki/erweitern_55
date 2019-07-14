import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint

class Network:
    def __init__(self):
        # Input layer
        input_image = keras.layers.Input(shape=[5, 5, 68], dtype='float32')

        # Convolution layer
        x = keras.layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu)(input_image)

        # Residual blocks
        for i in range(5):
            x = self.residual_block(x)

        # Policy head
        policy = keras.layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu)(x)
        policy = keras.layers.Conv2D(69, [3, 3], padding='same', activation=tf.nn.relu)(policy)
        policy = keras.layers.Flatten()(policy)
        policy = keras.layers.Softmax(name='policy')(policy)

        # Value head
        value = keras.layers.Conv2D(1, [3, 3], padding='same', activation=tf.nn.relu)(x)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(256, activation=tf.nn.relu)(value)
        value = keras.layers.Dense(1, activation=tf.nn.tanh, name='value')(value)

        # 入力層、出力層を定義
        self.model = keras.Model(inputs=input_image, outputs=[policy, value])

        # optimizerを定義
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9),
                    loss={'policy': keras.losses.categorical_crossentropy,
                          'value': keras.losses.mean_squared_error},
                    loss_weights={'policy': 1, 'value': 1})

    def residual_block(self, input_image, conv_kernel_shape=[3, 3]):
        conv_filters = int(input_image.shape[3])

        x = keras.layers.Conv2D(conv_filters, conv_kernel_shape, padding='same')(input_image)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(conv_filters, conv_kernel_shape, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, input_image])
        x = keras.layers.Activation('relu')(x)

        return x

    def step(self, train_images, policy_label, value_label):
        loss = self.model.train_on_batch(train_images, [policy_label, value_label])
        return loss

    def predict(self, images):
        return self.model.predict(images, batch_size=len(images), verbose=1, steps=None)
