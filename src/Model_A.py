
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = (64, 64)
NUM_CLASSES = 5
WEIGHT_DECAY = 1e-4


def conv_block(x, filters, dropout_rate=0.0):
    # Two convolution layers for stronger feature extraction
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_model_a():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    # Normalize RGB values from [0, 255] to [0, 1]
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = conv_block(x, 32, dropout_rate=0.05)
    x = conv_block(x, 64, dropout_rate=0.10)
    x = conv_block(x, 128, dropout_rate=0.15)

    # Compact classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions",
    )(x)

    model = models.Model(inputs, outputs, name="model_a_improved")

    return model


if __name__ == "__main__":
    model = build_model_a()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
