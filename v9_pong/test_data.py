import pickle
import gym
import gym.envs.atari
import random
env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
train_data = pickle.load(open("datas/train_pong.p","rb"))

env.reset()

state, r_best, best_as = random.choice(train_data)

print r_best

env.restore_full_state(state)
for aa in best_as:
  # okayz = raw_input('type any key') 
  obs, reward, done, info = env.step(aa)
  print reward
  env.render()

