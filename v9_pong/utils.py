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
    obs = preprocess(obs)
    return _get_rectangle(obs, 92)

# Retrieves the rectangle representing the ball.
def get_ball(obs):
    obs = preprocess(obs)
    return _get_rectangle(obs, 236)

def same_line_print(message):
  sys.stdout.write("\r" + message)
  sys.stdout.flush()

def render_state(env, state):
  env.reset()
  env.restore_full_state(state)
