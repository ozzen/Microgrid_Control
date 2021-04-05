"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: Shouvik Roy
Description: Implementing DDPG algorithm on the Microgrid Control Problem.
"""

"""
1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value) or
 bad (negative value) given a state and an action.

DDPG uses two more techniques not present in the original DQN:

**First, it uses two Target networks.**

**Why?** Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable.

**Second, it uses Experience Replay.**

We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.
"""

import os
import math
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

num_states = 3
print("Size of State Space ->  {}".format(num_states))
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))

init_v = -0.3
high_v = 0.5
high_f = 0
low_f = 0
angle = 0
n = 2
w = 100
dev = 0.02
fsc = 0.15
volt_ref = 0

upper_bound = 1.0
lower_bound = -1.0

def bound(n, minn, maxn):
    return max(min(maxn, n), minn)

def reset():
    # Initial Configurations of DDPG-based agent DER
    a_init = np.random.uniform(low=-angle,high=angle)
    f_init = np.random.uniform(low=low_f,high=high_f)
    v_init = np.random.uniform(low=-high_v,high=init_v)

    if np.sign(a_init) != np.sign(v_init):
        a_init = -a_init

    state = np.array([a_init,f_init,v_init])
    return state

def step_1(state, action, neigh):
    u = action[0]

    z1 = state[0]
    z2 = state[1]
    z3 = state[2]

    z4 = neigh[0]
    z5 = neigh[1]
    z6 = neigh[2]

    ang = z3 - z2*z3
    freq = z1 * z3
    volt = u + 33.6*z2 - 67.8*z1 - 1.89*z3 + 16.9715*z4 + 1.9718*z5 - 1.9718*z1*z4 + 16.9715*z1*z5 - 16.9715*z2*z4 - 1.9718*z2*z5

    if np.sign(ang) != np.sign(volt):
        ang = -ang
    ang = bound(ang, -angle, angle)
    freq = bound(freq, low_f, high_f)
    volt = bound(volt, -high_v, high_v)

    next_state = np.array([ang,freq,volt])
    return next_state

def step_2(state, action, neigh):
    u = action[0]

    z1 = neigh[0]
    z2 = neigh[1]
    z3 = neigh[2]

    z4 = state[0]
    z5 = state[1]
    z6 = state[2]

    ang = z6 - z5*z6
    freq = z4 * z6
    volt = u + 11.3986*z1 + 1.2088*z2 - 98.8604*z4 + 48.4810*z5 - 1.2658*z6 - 1.2088*z1*z4 - 11.3986*z1*z5 + 11.3986*z2*z4 - 1.2088*z2*z5

    if np.sign(ang) != np.sign(volt):
        ang = -ang
    ang = bound(ang, -angle, angle)
    freq = bound(freq, low_f, high_f)
    volt = bound(volt, -high_v, high_v)

    next_state = np.array([ang,freq,volt])
    return next_state

def get_reward(node1, node2):
    volt_n1 = node1[2]
    volt_n2 = node2[2]

    if volt_n1 <= volt_ref + dev and volt_n1 >= volt_ref - dev:
        reward = 100
    # elif volt_n1 <= volt_ref - fsc or volt_n1 >= volt_ref + fsc:
    #     reward = -50
    else:
        reward = tanh_sq(volt_n1, volt_n2)

    return reward

def neg_sq(v1, v2):
    r = - ((v1 - volt_ref)**2) - ((v2 - volt_ref)**2)
    R = (w/n) * r
    return R

def tanh_sq(v1, v2):
    r = - (math.tanh(abs(v1 - volt_ref))**2) - (math.tanh(abs(v2 - volt_ref))**2)
    R = (w/n) * r
    return R

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically an **Ornstein-Uhlenbeck process** for generating noise, as described 
in the paper. It samples noise from a correlated normal distribution.
"""

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

"""
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=128):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input)
    state_out = layers.Dense(128, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(128, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

"""
`policy()` returns an action sampled from our Actor network plus some noise for exploration.
"""

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

"""
## Training hyperparameters
"""

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
actor_lr = 0.0001
critic_lr = 0.0001

actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

actor_model.compile(optimizer=actor_optimizer)
critic_model.compile(optimizer=critic_optimizer)
target_actor.compile(optimizer=actor_optimizer)
target_critic.compile(optimizer=critic_optimizer)

total_episodes = 2000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001

buffer = Buffer(50000, 128)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

ep_reward_list = []
avg_reward_list = []
voltage_list_n1 = []
voltage_list_n2 = []

for ep in range(total_episodes):
    prev_state_n1 = reset()
    prev_state_n2 = reset()
    episodic_reward_n1 = 0
    episodic_reward_n2 = 0
    count = 0

    while count < 200:
        tf_prev_state_n1 = tf.expand_dims(tf.convert_to_tensor(prev_state_n1), 0)
        tf_prev_state_n2 = tf.expand_dims(tf.convert_to_tensor(prev_state_n2), 0)

        action_n1 = policy(tf_prev_state_n1, ou_noise)
        action_n2 = policy(tf_prev_state_n2, ou_noise)

        # Recieve state and reward from environment.
        state_n1 = step_1(prev_state_n1, action_n1, prev_state_n2)
        state_n2 = step_2(prev_state_n2, action_n2, prev_state_n1)

        reward_n1 = get_reward(state_n1, state_n2)
        reward_n2 = get_reward(state_n2, state_n1)
        # print(state_n1, action_n1, reward_n1, state_n2, action_n2, reward_n2)

        buffer.record((prev_state_n1, action_n1, reward_n1, state_n1))
        buffer.record((prev_state_n2, action_n2, reward_n2, state_n1))

        episodic_reward_n1 += reward_n1
        episodic_reward_n2 += reward_n2

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        prev_state_n1 = state_n1
        prev_state_n2 = state_n2

        count = count + 1

    ep_reward_list.append(episodic_reward_n1)
    avg_reward_n1 = np.mean(ep_reward_list[-500:])
    if ep % 100 == 0:
        print("Episode * {} * Avg Reward is ==> {} and Epsiodic Reward is ==> {}".format(ep, avg_reward_n1, episodic_reward_n1))
    avg_reward_list.append(avg_reward_n1)

parent_dir = "/Users/admin/Desktop/DDPG"
result_dir = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M')
result_path = os.path.join(parent_dir, result_dir)
os.mkdir(result_path)
data_dir = "data"
plot_dir = "plots"
data_path = os.path.join(result_path,data_dir)
plot_path = os.path.join(result_path,plot_dir)
os.mkdir(data_path)
os.mkdir(plot_path)

# Collecting data
# df1 = pd.DataFrame(voltage_list_n0)
# df1.to_csv(os.path.join(data_path,'Voltage_n0.csv'),index=False)
# df2 = pd.DataFrame(voltage_list_n1)
# df2.to_csv(os.path.join(data_path,'Voltage_n1.csv'),index=False)
# df3 = pd.DataFrame(voltage_list_n2)
# df3.to_csv(os.path.join(data_path,'Voltage_n2.csv'),index=False)
df4 = pd.DataFrame(avg_reward_list)
df4.to_csv(os.path.join(data_path,'Avg_Reward.csv'),index=False)
df5 = pd.DataFrame(ep_reward_list)
df5.to_csv(os.path.join(data_path,'Ep_Reward.csv'),index=False)

# Plotting graphs
# Episodes versus Average Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Reward")
plt.savefig(os.path.join(plot_path,"Avg_Reward_plot.png"))
plt.show()

# Episodes versus Episodic Rewards
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Ep. Reward")
plt.savefig(os.path.join(plot_path,"Ep_Reward_plot.png"))
plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.
"""

# Save the model
actor_model.save(os.path.join(result_path,"microgrid_actor.h5"))
critic_model.save(os.path.join(result_path,"microgrid_critic.h5"))
actor_model.save_weights(os.path.join(result_path,"microgrid_actor_weights.h5"))
critic_model.save_weights(os.path.join(result_path,"microgrid_critic_weights.h5"))

target_actor.save(os.path.join(result_path,"microgrid_target_actor.h5"))
target_critic.save(os.path.join(result_path,"microgrid_target_critic.h5"))
target_actor.save_weights(os.path.join(result_path,"microgrid_target_actor_weights.h5"))
target_critic.save_weights(os.path.join(result_path,"microgrid_target_critic_weights.h5"))