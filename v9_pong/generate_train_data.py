import gym
import gym.envs.atari
import random
import sys

num_states = 100

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=3)

def same_line_print(message):
  sys.stdout.write("\r" + message)
  sys.stdout.flush()

def get_random_states(start_state):
  env.reset()
  env.restore_full_state(start_state)
  for i in range(random.randint(100, 500)):
    _, a, b, c = env.step(random.choice([2,3]))
  state = env.clone_full_state()
  return state

env.reset()
start_state = env.clone_full_state()

states = [get_random_states(start_state) for i in range(num_states)]

# for ss in states:
#   print ss
#   env.restore_full_state(ss)
#   okayz = raw_input('type any key') 
#   env.step(3)
#   env.render()

def get_trace(start_state):
  env.reset()
  env.restore_full_state(start_state)

  rewards = 0.0
  actions = []
  for i in range(100):
    action = random.choice([2,3])
    observation, reward, done, info = env.step(action)
    rewards += reward
    actions.append(action)
  
  return rewards, actions

def estimate_state_value(start_state):
  n_sample = 1000
  reward = 0.0 
  for i in range(n_sample):
    same_line_print(str(i) + " out of 1000")
    rs, asss = get_trace(start_state)
    reward += rs
  return reward / n_sample

def get_good_trace(start_state):
  best_reward = -9999
  best_actions = []
  for i in range(1000):
    same_line_print(str(i) + " out of 1000")
    rs, asss = get_trace(start_state)
    if rs > best_reward:
      best_reward = rs
      best_actions = asss
  print "best reward ", best_reward
  return best_reward, best_actions

def get_good_action(start_state):
  move1 = 2
  move2 = 3
  
  env.reset()
  env.restore_full_state(start_state)

  env.step(move1)
  state1 = env.clone_full_state()
  
  env.reset()
  env.restore_full_state(start_state)

  env.step(move2)
  state2 = env.clone_full_state()

  return start_state, (move1, estimate_state_value(state1)),\
                      (move2, estimate_state_value(state2))

to_store = []

state_number = 0
for state in states:
  state_number += 1
  print "solving for state number ", state_number
  good_action_result = get_good_action(state)
  print good_action_result
  to_store.append(good_action_result)
  

import pickle
pickle.dump( to_store, open( "datas/train_pong.p", "wb" ) )


