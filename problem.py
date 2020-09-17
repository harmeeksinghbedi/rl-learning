import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from Actor  import Actor
from Critic import Critic 
from Noise  import OUActionNoise, OUNoise
from Buffer import Buffer

problem="MountainCarContinuous-v0"

env = gym.make(problem)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print("Size of State Space ->  {}".format(num_states))
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

#Size of State Space ->  2
#Size of Action Space ->  1
#Max Value of Action ->  1.0
#Min Value of Action ->  -1.0

# returns a action sampled from our Actor adding noise
def get_policy_action(current_state, actor_model, noise_object):

  # get sampled actions
  actions = actor_model(current_state)
  pactions = actions 
  noise = noise_object.sample()

  # Add noise to action  
  actions = actions.numpy() + noise
  pureactions = pactions.numpy()
  # We make sure action is within bounds
  pure_actions = np.clip(pureactions, lower_bound, upper_bound)
  legal_action = np.clip(actions, lower_bound, upper_bound)
  return [np.squeeze(legal_action)], [np.squeeze(pure_actions)]


std_dev = 0.25
exploration_mu = 0
exploration_theta = 0.05
exploration_sigma = std_dev

#ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
ou_noise = OUNoise(num_actions, exploration_mu, exploration_theta, exploration_sigma)

actor_model = Actor(num_states, upper_bound)
actor_model = actor_model.get_model()
critic_model = Critic(num_states, num_actions)
critic_model = critic_model.get_model()

target_actor_model = Actor(num_states, upper_bound)
target_actor_model = target_actor_model.get_model()
target_critic_model = Critic(num_states, num_actions)
target_critic_model = target_critic_model.get_model()

# Making the weights equal initially
target_actor_model.set_weights(actor_model.get_weights())
target_critic_model.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 500
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
buffer = Buffer(num_states, num_actions, 50000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# loop over all episodes 
for ep in range(total_episodes):
   prev_state = env.reset()
   episodic_reward = 0

   while True: 

      # render environment
      env.render()

      tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
 
      # get a noisy action (for exploration)
      action, pureaction = get_policy_action(tf_prev_state, actor_model, ou_noise)

      # take a step 
      state , reward, done , info = env.step(action)

      buffer.remember_experience((prev_state, action, reward, state))
      episodic_reward += reward

      # process and learn
      buffer.process_batch(gamma, critic_model, target_critic_model, actor_model, target_actor_model, critic_optimizer, actor_optimizer) 
      buffer.update_target(tau, critic_model, target_critic_model, actor_model, target_actor_model)

      if done:
        break

      prev_state = state 

   ep_reward_list.append(episodic_reward)

   #Mean of last 40 episodes
   avg_reward = np.mean(ep_reward_list[-40:])
   print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
   avg_reward_list.append(avg_reward)

actor_model.save_weights("car_actor.h5")
critic_model.save_weights("car_critic.h5")

target_actor_model.save_weights("car_target_actor.h5")
target_critic_model.save_weights("car_target_critic.h5")

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
