
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (64, 64)
NUM_CLASSES = 5


def build_model_a():
    model = models.Sequential(name="model_a_standard")
    model.add(layers.Input(shape=(*IMG_SIZE, 3)))

    for filters in [32, 64, 128]:
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D(2))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model


if __name__ == "__main__":
    model = build_model_a()
    model.summary()
