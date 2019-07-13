import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint

class Network:
    def __init__(self):
        # Input layer
        input_image = keras.layers.Input(shape=[5, 5, 266], dtype='float32')

        # Convolution layer
        x = keras.layers.Conv2D(256, [3, 3], padding='same', activation=tf.nn.relu)(input_image)

        # Residual blocks
        for i in range(9):
            x = self.residual_block(x)

        # Policy head
        policy = keras.layers.Conv2D(256, [3, 3], padding='same', activation=tf.nn.relu)(x)
        policy = keras.layers.Conv2D(69, [3, 3], padding='same', activation=tf.nn.relu)(policy)
        policy = keras.layers.Softmax(name='policy')(policy)

        # Value head
        value = keras.layers.Conv2D(1, [3, 3], padding='same', activation=tf.nn.relu)(x)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(1, activation=tf.nn.tanh, name='value')(value)

        # 入力層、出力層を定義
        self.model = keras.Model(inputs=input_image, outputs=[policy, value])

        # optimizerを定義
        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9),
                    loss={'policy': keras.losses.CategoricalCrossentropy(from_logits=True),
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

    def train(self, train_images, policy_label, value_label, batch_size, epochs):
        # epochによってlearning rateを変更する
        learning_rate_scheduler = LearningRateScheduler(lambda epoch: 0.01 if epoch==0 else 0.001 if epoch==1 else 0.0001,
                                                        verbose=True)

        # epochごとに重みを保存
        save_path = './weights/epoch_{epoch:02d}.hdf5'
        model_check_point = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=False)

        self.model.fit(train_images, [policy_label, value_label], batch_size=batch_size, epochs=epochs, callbacks=[model_check_point, learning_rate_scheduler])

    def accuracy(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

    def predict(self, images):
        return self.model.predict(images, verbose=1, steps=None)
