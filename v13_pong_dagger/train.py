import tensorflow as tf
import pickle
import gym
import gym.envs.atari
import random
import numpy as np
from utils import get_signal

out_dim = 2
input_dim = 9
n_batch = 30

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)

train_data = pickle.load(open("datas/train_tensors.p","rb"))

def get_train_batch(n_batch, train_data):
  obss = []
  actions = []
  for i in range(n_batch):
    _obss, _action = random.choice(train_data)
    obss.append(_obss)
    actions.append(_action)
  return np.array(obss), np.array(actions)

# this is the input state
input_state = tf.placeholder(tf.float32, [None, input_dim])
output_state = tf.placeholder(tf.float32, [None, out_dim])

num_hidden = 4

# predict the move
hidden = tf.layers.dense(input_state, num_hidden)
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

def act(obs, prev_obs, prev_move):
  
  signal = get_signal(obs, prev_obs, prev_move)
  accion = session.run([pred_prob], {input_state: np.array([signal])})[0][0]
  print signal, accion
  xxx = np.argmax(accion)
  action = np.random.choice([2,3], p=accion)
  # action = [2,3][xxx]
  return action

verify_envs = pickle.load(open("datas/train_pong.p","rb"))

# take a random state and run for 300 iteration to see what would happen
def random_test(state):
  env.reset()
  env.restore_full_state(state)
  obss = [None, None]
  movez = [2]
  for i in range(300):
    action = act(obss[-1], obss[-2], movez[-1])
    obs, reward, done, info = env.step(action)
    env.render()
    obss.append(obs)
    movez.append(action)

for i in range(1000000):
  print i 
  obss, action = get_train_batch(n_batch, train_data)
  session.run([train], {input_state: obss, output_state: action})

  if i % 1000 == 0:
    state, _blah, _blah2 = random.choice(verify_envs)
    random_test(state)

