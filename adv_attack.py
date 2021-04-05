import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

actor = load_model('/Users/admin/Desktop/DDPG/mg_diesel/results/3/microgrid_actor.h5')
critic = load_model('/Users/admin/Desktop/DDPG/mg_diesel/results/3/microgrid_critic.h5')

# Attack parameters
k = 100
alpha = 2
beta = 10
runs = 100
epsilon = 0.1
j = 0

# Microgrid Parameters
n = 2
w = 100
init_v = 0.0
high_v = 0.5
high_f = 0.0
angle = 0.0
volt_ref = 0.0
dev = 0.01

def bound(N, minn, maxn):
    return max(min(maxn, N), minn)

def reset_battery():
    v_start = np.random.uniform(low=-high_v,high=high_v)
    f_start = np.random.uniform(low=-high_f,high=high_f)
    der_angle = angle * f_start
    state = [der_angle,f_start,v_start]
    return state

def reset_diesel():
    # Initial Configurations of DDPG-based agent DER
    a_init = np.random.uniform(low=-angle,high=angle)
    f_init = np.random.uniform(low=-high_f,high=high_f)
    v_init = np.random.uniform(low=-high_v,high=high_v)

    # if np.sign(a_init) != np.sign(v_init):
    #     a_init = -a_init

    state = np.array([a_init,f_init,v_init])
    return state

def next_battery(prev_state, action, prev_neigh1, prev_neigh2):
    u = action[0]
    u = u[0]

    v0 = prev_state[0]
    f0 = prev_state[1]
    a0 = prev_state[2]

    v1 = prev_neigh1[0]
    f1 = prev_neigh1[1]
    a1 = prev_neigh1[2]

    v2 = prev_neigh2[0]
    f2 = prev_neigh2[1]
    a2 = prev_neigh2[2]

    ap1 = v0 * (v1 * (math.cos(a1) - math.sin(a1)) + v2 * (math.cos(a2) - math.sin(a2)))
    volt = u + 0.3 * (v1 + v2 - ap1)
    volt = bound(volt, -high_v, high_v)

    ap2 = v0 * (v1 * (-math.sin(a1) - math.cos(a1)) + v2 * (-math.sin(a2) - math.cos(a2)))
    freq = f0 + 0.3 * (-ap2)
    freq = bound(freq, -high_f, high_f)

    ang = a0 * freq

    next = np.array([volt,freq,ang], dtype=np.float)
    ip_v = next[0]
    ip_f = next[1]
    ip_a = next[2]
    next_state = [ip_v, ip_f, ip_a]
    return next_state

def next_diesel_1(state, action, neigh):
    u = action[0]
    u = u[0]

    z1 = state[0]
    z2 = state[1]
    z3 = state[2]

    z4 = neigh[0]
    z5 = neigh[1]
    z6 = neigh[2]

    ang = z3 - z2*z3
    freq = z1 * z3
    volt = u + 33.6*z2 - 67.8*z1 - 1.89*z3 + 16.9715*z4 + 1.9718*z5 - 1.9718*z1*z4 + 16.9715*z1*z5 - 16.9715*z2*z4 - 1.9718*z2*z5

    ang = bound(ang, -angle, angle)
    freq = bound(freq, -high_f, high_f)
    volt = bound(volt, -high_v, high_v)

    next = np.array([ang, freq, volt], dtype=np.float)
    ip_a = next[0]
    ip_f = next[1]
    ip_v = next[2]
    next_state = [ip_a, ip_f, ip_v]
    return next_state

    next_state = np.array([ang,freq,volt])
    return next_state

def next_diesel_2(state, action, neigh):
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

    ang = bound(ang, -angle, angle)
    freq = bound(freq, -high_f, high_f)
    volt = bound(volt, -high_v, high_v)

    next = np.array([ang, freq, volt], dtype=np.float)
    ip_a = next[0]
    ip_f = next[1]
    ip_v = next[2]
    next_state = [ip_a, ip_f, ip_v]
    return next_state

    next_state = np.array([ang,freq,volt])
    return next_state

def adv_attack(state):
    state_adv = [0,0,0]
    Q_value = get_Q(state_n0, state_n1)
    theta = state[0]
    freq = state[1]
    volt = state[2]
    # print(state_n0, Q_value)
    for i in range(1,k+1):
        noise = np.random.beta(alpha, beta) - 0.5
        adv_volt = volt + epsilon * noise
        # adv_freq = freq + epsilon * noise
        # adv_theta = theta + epsilon * noise
        new_state = [theta, freq, adv_volt]
        adv_state = tf.expand_dims(tf.convert_to_tensor(new_state), 0)
        action_adv = actor.predict(adv_state)
        next_state = next_diesel_1(new_state, action_adv, state_n1)
        Q_value_adv = get_Q(next_state, state_n1)
        if Q_value_adv < Q_value:
            Q_value = Q_value_adv
            state_adv = new_state
        # print(new_state, Q_value_adv)
    return state_adv

def get_Q(state_n0, state_n1):
    v0 = state_n0[2]
    v1 = state_n1[2]
    # v2 = state_n2[0]
    reward = (w/n) * (- ((v0 - volt_ref)**2) - ((v1 - volt_ref)**2))# - ((v2 - volt_ref)**2))
    Q = 0.99 * reward
    return Q

for i in range(1,runs+1):
    # Get initial state
    state_n0 = reset_diesel()
    state_n1 = reset_diesel()
    # state_n2 = reset_battery()

    # Get current state
    curr_state_n0 = tf.expand_dims(tf.convert_to_tensor(state_n0), 0)
    curr_state_n1 = tf.expand_dims(tf.convert_to_tensor(state_n1), 0)
    # curr_state_n2 = tf.expand_dims(tf.convert_to_tensor(state_n2), 0)

    # Get current action
    action_n0 = actor.predict(curr_state_n0)
    action_n1 = actor.predict(curr_state_n1)
    # action_n2 = actor.predict(curr_state_n2)

    # Get next state
    next_state_n0 = next_diesel_1(state_n0, action_n0, state_n1)
    next_state_n1 = next_diesel_2(state_n1, action_n1, state_n0)
    # next_state_n2 = next(state_n2, action_n2, state_n0, state_n1)

    # Get adversarial state
    adversary = adv_attack(state_n0)

    if adversary[0] != 0 or adversary[1] != 0 or adversary[2] != 0:
        # Store all states
        df1 = pd.DataFrame(adversary)
        df1 = pd.DataFrame.transpose(df1)
        df2 = pd.DataFrame(state_n1)
        df2 = pd.DataFrame.transpose(df2)
        # df3 = pd.DataFrame(state_n2)
        # df3 = pd.DataFrame.transpose(df3)
        frames = [df1, df2]
        result = pd.concat(frames)
        j = j + 1
        result.to_csv('mg_diesel/adv/Adversarial_States_{}.csv'.format(j),index=False)

        # Clear dataframes
        df1.drop(df1.index, inplace=True)
        df2.drop(df2.index, inplace=True)
        # df3.drop(df3.index, inplace=True)

        print("Success for run {}".format(i))
    else:
        print("Failure for run {}".format(i))
