import tensorflow as tf
import pickle
import gym
import gym.envs.atari
import random
import numpy as np
from script_trace_to_tensors import get_signal

out_dim = 3
input_dim = 6

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=3)

train_data = pickle.load(open("datas/train_tensors.p","rb"))

# this is the input state
input_state = tf.placeholder(tf.float32, [None, input_dim])
output_state = tf.placeholder(tf.float32, [None, out_dim])

num_hidden = 100

# predict the move
hidden = tf.layers.dense(input_state, 100)
prediction = tf.layers.dense(hidden, out_dim)
pred_prob = tf.nn.softmax(prediction)
pred_prob = pred_prob + 1e-8

# set up the cost function for training
log_pred_prob = tf.log(pred_prob)
objective = tf.reduce_mean(log_pred_prob * output_state)
    
loss = -objective

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

initializer = tf.global_variables_initializer()

session = tf.Session()
session.run(initializer)

for i in range(10000):
  print i 
  obss, action = random.choice(train_data)
  session.run([train], {input_state: np.array([obss]),
                        output_state: np.array([action])})

def act(obs, prev_obs):
  
  signal = get_signal(obs, prev_obs)
  accion = session.run([pred_prob], {input_state: np.array([signal])})[0][0]
  print signal, accion
  xxx = np.argmax(accion)
  action = np.random.choice([0,2,3], p=accion)
  # action = [0,2,3][xxx]
  return action

verify_envs = pickle.load(open("datas/train_pong.p","rb"))
for state, r_best, best_as in verify_envs:
  env.reset()
  obs = env.restore_full_state(state)
  done = False
  prev_obs = None
  for i in range(100):
    action = act(obs, prev_obs)
    obs, reward, done, info = env.step(action)
    env.render()
    prev_obs = obs
