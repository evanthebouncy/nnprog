import numpy as np
import gym
import random
import pickle
import gym.envs.atari

# Preprocesses the given image:
# (1) remove the scoreboard
# (2) make it monochromatic
# (3) make the background black
#
# obs: Image
# return: Image
# Image = np.array([n_rows, n_cols])
def preprocess(obs):
    obs = obs[34:194]
    obs = obs[::2,::2,0]
    obs[obs == 144] = 0
    return obs.astype(np.float)


# Assumes that the pixels of the given value in the given image
# exactly form a rectangle (or else there are no pixels of that color).
# Returns the rectangle if it exists, or else None.
#
# val: int
# obs: Image
# return: None | Rectangle
# Image = np.array([n_rows, n_cols])
def _get_rectangle(obs, val):
    min_val = np.argmax(obs.ravel() == val)
    max_val = len(obs.ravel()) - np.argmax(np.flip(obs.ravel(), 0) == val) - 1
    x_pos = min_val % obs.shape[1]
    y_pos = min_val / obs.shape[1]
    x_len = (max_val % obs.shape[1]) - x_pos + 1
    y_len = (max_val / obs.shape[1]) - y_pos + 1
    return None if x_pos == 0 and y_pos == 0 and x_len == obs.shape[1] and y_len == obs.shape[0] else np.array([x_pos + x_len/2, y_pos + y_len/2])

# Retrieves the rectangle representing our paddle.
def get_our_paddle(obs):
    return _get_rectangle(obs, 92)

# Retrieves the rectangle representing the ball.
def get_ball(obs):
    return _get_rectangle(obs, 236)

def get_signal(obs, prev_obs):
  default_val = np.array([0.0 for i in range(6)])
  if obs == None or prev_obs == None: 
    return default_val

  obs = preprocess(obs)
  paddle = get_our_paddle(obs)
  ball = get_ball(obs)
  prev_ball = get_ball(prev_obs)

  if ball == None or paddle == None:
    return default_val

  diff = ball - prev_ball
  diff = diff / np.max(abs(diff))

  signal = np.concatenate([paddle, ball, diff]) / 80.0
  return signal

def one_hot(num):
  if num == 0: return [1.0, 0.0, 0.0]
  if num == 2: return [0.0, 1.0, 0.0]
  if num == 3: return [0.0, 0.0, 1.0]
  assert 0

if __name__ == "__main__":

  train_tensors = []
  env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=3)
  train_data = pickle.load(open("datas/train_pong.p","rb"))

  prev_obs = None

  for state, r_best, best_as in train_data:
    env.reset()
    env.restore_full_state(state)
    for aa in best_as:
      # okayz = raw_input('type any key') 
      obs, reward, done, info = env.step(aa)
      signal = get_signal(obs, prev_obs)
      print signal, one_hot(aa)
      train_tensors.append((signal, one_hot(aa)))
      prev_obs = obs

  print train_tensors
  print len(train_tensors)

  import pickle
  pickle.dump(train_tensors, open( "datas/train_tensors.p", "wb" ) )
