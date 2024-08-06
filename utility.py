import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.activations import linear, relu, sigmoid


def build_models():
    tf.random.set_seed(1234)  # for consistent result
    model1 = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(units=25, activation="relu"),
            Dense(units=15, activation="relu"),
            Dense(units=10, activation="linear"),
        ],name = "model1"
    )
    model2 = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(units=75, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=50, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=30, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=18, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=10, activation="linear"),
        ],name = "model2"
    )
    model3 = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(units=90, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=80, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=50, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=35, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=23, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=13, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
            Dense(units=10, activation="linear"),
        ],name = "model3"
    )
    model4 = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(units=300, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=259, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=200, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=180, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=149, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=90, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=79, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=40, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            Dense(units=10, activation="linear"),
        ],name = "model4"
    )

    model_list = [model1,model2,model3,model4]
    return model_list