import gym
import gym.envs.atari
import random
env = gym.envs.atari.atari_env.AtariEnv(obs_type='image')

def get_random_states(start_state):
  env.reset()
  env.restore_full_state(start_state)
  for i in range(random.randint(100, 500)):
    _, a, b, c = env.step(random.choice([2,3]))
  state = env.clone_full_state()
  return state

env.reset()
start_state = env.clone_full_state()

states = [get_random_states(start_state) for i in range(10)]

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
    action = random.choice([2,3,0])
    observation, reward, done, info = env.step(action)
    rewards += reward
    actions.append(action)
  
  return rewards, actions


def get_good_trace(start_state):
  best_reward = -9999
  best_actions = []
  for i in range(1000):
    rs, asss = get_trace(states[0])
    if rs > best_reward:
      best_reward = rs
      best_actions = asss

  return best_reward, best_actions

r_best, best_as = get_good_trace(states[0])

print r_best, best_as

env.reset()
env.restore_full_state(states[0])
for aa in best_as:
  okayz = raw_input('type any key') 
  env.step(aa)
  env.render()
        
