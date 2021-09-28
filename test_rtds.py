import time
import pandas as pd
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# load actor network
actor = load_model('/Users/admin/Desktop/DDPG/mg_rtds/9/microgrid_actor.h5')

# starting matlab engine
eng = matlab.engine.start_matlab()

# rtds constraints
high_i_d = 0.7
low_i_d = 0.6
high_v_d = 1.0
high_q = 0.0001
vd_ref = 0.48
vd_dev = 0.01
vd_fsc = 0.0144
w = 100
n = 1
dur = 0
j = 1

# contain values within a range
def bound(n, minn, maxn):
    return max(min(maxn, n), minn)

# generates initial configuration
def reset():
    i_d = np.random.uniform(low=low_i_d, high=high_i_d)  # d-current
    i_q = np.random.uniform(low=-high_q, high=high_q)  # q-current
    i_od = np.random.uniform(low=low_i_d, high=high_i_d)  # d-o/p of current controller
    i_oq = np.random.uniform(low=-high_q, high=high_q)  # q-o/p of current controller
    v_od = np.random.uniform(low=vd_ref - vd_fsc - 0.087, high=vd_ref + vd_fsc - 0.087)  # d-i/p to voltage controller
    v_oq = np.random.uniform(low=0.006, high=0.008)  # q-i/p to voltage controller
    i_ld = np.random.uniform(low=low_i_d, high=high_i_d)  # d-i/p to current controller
    i_lq = np.random.uniform(low=-high_q, high=high_q)  # q-i/p to current controller
    m_d = np.random.uniform(low=0.35, high=0.40)
    m_q = np.random.uniform(low=0.05, high=0.07)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    return state

# neural controller
def control(state, action):
    u_d = float(action[0])
    u_q = u_d
    i_d = float(state[0])
    i_q = float(state[1])
    i_od = float(state[2])
    i_oq = float(state[3])
    v_od = float(state[4])
    v_oq = float(state[5])
    i_ld = float(state[6])
    i_lq = float(state[7])
    m_d = float(state[8])
    m_q = float(state[9])

    next_state = eng.rtds_ode(i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q, u_d, u_q)

    i_d_n = next_state[0]
    i_q_n = next_state[1]
    i_od_n = next_state[2]
    i_oq_n = next_state[3]
    v_od_n = next_state[4]
    v_oq_n = next_state[5]
    i_ld_n = next_state[6]
    i_lq_n = next_state[7]
    m_d_n = next_state[8]
    m_q_n = next_state[9]

    i_d_n = bound(i_d_n, low_i_d, high_i_d)
    i_q_n = bound(i_q_n, -high_q, high_q)
    i_od_n = bound(i_od_n, low_i_d, high_i_d)
    i_oq_n = bound(i_oq_n, -high_q, high_q)
    v_od_n = bound(v_od_n, 0, high_v_d)
    v_oq_n = bound(v_oq_n, 0.006, 0.008)
    i_ld_n = bound(i_ld_n, low_i_d, high_i_d)
    i_lq_n = bound(i_lq_n, -high_q, high_q)
    m_d_n = bound(m_d_n, 0.35, 0.40)
    m_q_n = bound(m_q_n, 0.05, 0.07)

    next_state = [i_d_n, i_q_n, i_od_n, i_oq_n, v_od_n, v_oq_n, i_ld_n, i_lq_n, m_d_n, m_q_n]

    return next_state

# load adversarial states
def adversary(runs):
    df = pd.read_csv('/Users/admin/Desktop/DDPG/mg_rtds/adv/adv_init/Adversarial_States_{}.csv'.format(runs))
    df_list = df.values.tolist()
    return df_list[0]

# Voltage_List = []

for runs in range(1,j+1):
    timesteps = 0
    prev_state = reset()
    # prev_state = adversary(runs)
    voltage_list_d = [prev_state[4]]
    while timesteps < 3124:
        # generate state, action and next state
        state_tf = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        start = time.perf_counter()
        action = actor.predict(state_tf)
        end = time.perf_counter()
        dur = dur + (end - start)
        state = control(prev_state, action)
        prev_state = state

        # store d-component of voltage
        voltage_d = state[4]
        voltage_list_d.append(voltage_d)
        # Voltage_List.append(voltage_d)
        timesteps = timesteps + 1
        print("Number of time steps:", timesteps)
        # print(prev_state)

    # generate outputs
    df1 = pd.DataFrame(voltage_list_d)
    # df1.to_csv('/Users/admin/Desktop/DDPG/mg_rtds/Voltage_d_{}.csv'.format(runs),index=False)
    df1.drop(df1.index, inplace=True)
    # voltage_list_d = []
    print("Number of runs:", runs)

# closing matlab engine
eng.quit()

# df2 = pd.DataFrame(Voltage_List)
# df2.to_csv('/Users/admin/Desktop/DDPG/mg_rtds/3/tests/runs/Voltage.csv',index=False)

# compute execution time
dur = dur/(n*timesteps*(runs))
print("Time per action per agent in seconds:",dur)

# generate plots
r = np.full((timesteps,1), vd_ref-0.087)
plt.plot(voltage_list_d)
plt.plot(r, '--')
plt.xlabel("Time")
plt.ylabel("Voltage")
# plt.savefig('/Users/admin/Desktop/DDPG/mg_rtds/1/tests/Voltage_d_{}'.format(runs))
plt.show()
# plt.close()
