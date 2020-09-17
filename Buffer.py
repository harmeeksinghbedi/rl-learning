import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import Actor
import Critic

class Buffer:
   def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
      # Number of "experiences" to store at max
      self.buffer_capacity = buffer_capacity
      # Num of tuples to train on.
      self.batch_size = batch_size

      # Its tells us num of times record() was called.
      self.buffer_counter = 0

      # Instead of list of tuples as the exp.replay concept go
      # We use different np.arrays for each tuple element
      self.state_buffer      = np.zeros((self.buffer_capacity, num_states))
      self.action_buffer     = np.zeros((self.buffer_capacity, num_actions))
      self.reward_buffer     = np.zeros((self.buffer_capacity, 1))
      self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

   # stores (S,A,R,S') observations
   def remember_experience(self, observation_tuple):

      index = self.buffer_counter % self.buffer_capacity
      self.state_buffer[index] = observation_tuple[0]
      self.action_buffer[index] = observation_tuple[1]
      self.reward_buffer[index] = observation_tuple[2]
      self.next_state_buffer[index] = observation_tuple[2]
      self.buffer_counter += 1

   # process_batch 
   def process_batch(self, gamma, critic_model, target_critic_model, actor_model, target_actor_model, critic_optimizer, actor_optimizer):
     
     record_range = min(self.buffer_counter, self.buffer_capacity)
     batch_indexes = np.random.choice(record_range, self.batch_size)
      
     # get the random sample to learn against
     state_buffer  = tf.convert_to_tensor(self.state_buffer[batch_indexes])
     action_buffer = tf.convert_to_tensor(self.action_buffer[batch_indexes])
     reward_buffer = tf.convert_to_tensor(self.reward_buffer[batch_indexes])
     reward_buffer = tf.cast(reward_buffer, dtype=tf.float32)
     next_state_buffer = tf.convert_to_tensor(self.next_state_buffer[batch_indexes])

     # Train the Critic Model 
     with tf.GradientTape() as tape:
       # for Critic model training Compute y = r + gamma*Q_t1(s(t_1), a(t+1))
       a_t1 = target_actor_model(next_state_buffer)
       y_t  = reward_buffer + gamma * target_critic_model([next_state_buffer, a_t1])
       q_t1 = target_critic_model([next_state_buffer, a_t1])
       q_t  = critic_model([state_buffer, action_buffer]) 
       critic_loss = tf.math.reduce_mean(tf.math.square(y_t - q_t)) 
     critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
     critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

     # Train the Actor Model 
     with tf.GradientTape() as tape:
       # grad = MEAN(q(s_i, a_i)
       actions = actor_model(state_buffer)
       critic_value = critic_model([state_buffer, actions])

       # Used `-value` as we want to maximize the value given
       # by the critic for our actions
       actor_loss = -tf.math.reduce_mean(critic_value)
     actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
     actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

   # This update target parameters slowly
   # Based on rate `tau`, which is much less than one.
   def update_target(self, tau, critic_model, target_critic_model, actor_model, target_actor_model):
     new_weights = [] 
     target_variables = target_critic_model.weights
     for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

     target_critic_model.set_weights(new_weights)
     new_weights = []
     target_variables = target_actor_model.weights
     for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))
     target_actor_model.set_weights(new_weights)
      
