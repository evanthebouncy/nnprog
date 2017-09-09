import gym
import gym.envs.atari
import random
import sys

# =============== get keystroke logic ===============
class _Getch:
  """Gets a single character from standard input.  Does not echo to the
screen."""
  def __init__(self):
    try:
      self.impl = _GetchWindows()
    except ImportError:
      self.impl = _GetchUnix()

  def __call__(self): return self.impl()


class _GetchUnix:
  def __init__(self):
    import tty, sys

  def __call__(self):
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class _GetchWindows:
  def __init__(self):
    import msvcrt

  def __call__(self):
    import msvcrt
    return msvcrt.getch()

getch = _Getch()


num_states = 50

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

def get_move():
  while True:
    try:
      move = getch()
      move = int(move)
      assert move in [2,3], "move must be 2 3"
      return move
    except:
      pass

def get_trace(start_state):
  env.reset()
  env.restore_full_state(start_state)

  rewards = 0.0
  actions = []
  for i in range(200):
    env.render()
    action = get_move()
    observation, reward, done, info = env.step(action)
    rewards += reward
    actions.append(action)
  
  return start_state, rewards, actions

state_number = 0
to_store = []
for state in states:
  state_number += 1
  print "solving for state number ", state_number
  tr = get_trace(state)
  to_store.append(tr)

import pickle
pickle.dump( to_store, open( "datas/train_pong.p", "wb" ) )


