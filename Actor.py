import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


class Actor:
   def __init__(self, num_states, upper_bound):
     # Initialize weights between -3e-3 and 3-e3
     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

     inputs = layers.Input(shape=(num_states,))
     out = layers.Dense(512, activation="relu")(inputs)
     out = layers.BatchNormalization()(out)
     out = layers.Dense(512, activation="relu")(out)
     out = layers.BatchNormalization()(out)
     outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

     # Our upper bound is 1.0 for Car.
     outputs = outputs * upper_bound
     self.model = tf.keras.Model(inputs, outputs)

   def get_model(self):
     return self.model

