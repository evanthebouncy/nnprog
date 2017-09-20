import numpy as np
import gym
import random
import pickle
import gym.envs.atari
from utils import *


def one_hot(num):
  if num == 2: return [1.0, 0.0]
  if num == 3: return [0.0, 1.0]
  assert 0

if __name__ == "__main__":

  train_tensors = []
  env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
  train_data = pickle.load(open("datas/train_pong.p","rb"))

  for state, r_best, best_as in train_data:
    env.reset()
    env.restore_full_state(state)
    obss = [None, None]
    movez = [2]
    for aa in best_as:
      signal = get_signal(obss[-1], obss[-2], movez[-1])
      obs, reward, done, info = env.step(aa)
      # env.render()
      # signal = get_signal_full_image(obs, prev_obs)
      print signal, one_hot(aa)
      train_tensors.append((signal, one_hot(aa)))
      obss.append(obs)
      movez.append(aa)

  print len(train_tensors)

  import pickle
  pickle.dump(train_tensors, open( "datas/train_tensors.p", "wb" ) )
