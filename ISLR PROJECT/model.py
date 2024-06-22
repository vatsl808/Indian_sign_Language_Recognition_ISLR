import tensorflow as tf


def ISLR_MODEL(img_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(img_size, img_size, 1)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(27, activation='softmax')
    ])

    return model