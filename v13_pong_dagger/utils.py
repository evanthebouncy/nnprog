import numpy as np
import gym
import random
import pickle
import gym.envs.atari
import draw

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

def lose_color(proccessed_obs):
  return np.clip(proccessed_obs, 0, 1)


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

def get_simple_signal(ob1, ob2):
  default_val = np.array([1.0, 0.0, 0.0])
  if ob1 is None or ob2 is None: 
    return default_val

  # obs = preprocess(obs)
  paddle = get_our_paddle(ob2)
  ball = get_ball(ob2)

  if ball is None or paddle is None:
    return default_val

  return np.array([0.0, 1.0, 0.0]) if paddle[1] >= ball[1] else np.array([0.0, 0.0, 1.0])

def get_signal(obs, prev_obs, prev_move):
  
  default_val = np.array([0.0 for i in range(9)])
  if obs is None or prev_obs is None: 
    return default_val

  # obs = preprocess(obs)
  paddle = get_our_paddle(obs)
  ball = get_ball(obs)
  prev_ball = get_ball(prev_obs)
  prev_paddle = get_our_paddle(prev_obs)

  if ball is None or paddle is None or prev_ball is None or prev_paddle is None:
    return default_val

  # print "some stuff "
  # print "prev ball ", prev_ball
  # print "ball ", ball
  # print "paddle ", paddle

  # older
  paddle = paddle[1:] / 80.0
  prev_paddle = prev_paddle[1:] / 80.0
  diff = ball - prev_ball
  # print "diff ", diff
  diff = diff / float(np.max(abs(diff))) if np.max(abs(diff)) > 0 else np.array([0.0, 0.0])
  ball = ball / 80.0
  prev_move = np.array([1.0, 0.0] if prev_move == 2 else [0.0, 1.0])

  care = 1.0 if ball[0] >= 60.0 / 80.0 and ball[0] <= 71.0 / 80.0 else 0.0

  # print "ball ", ball

  signal = np.concatenate([paddle, prev_paddle, ball, diff, prev_move, [care]])
  signal = signal * care

  # newer
#  print ball, prev_ball
#  diff = ball - prev_ball
#  print "diff ", diff
#  a = 1.0 if paddle[1] > ball[1] else -1.0
#  b = diff[0]
#  signal = np.array([a,b, 0.0, 0.0, 0.0, 0.0])
  
  return signal

def get_signal_full_image(obs, prev_obs):
  if obs is None or prev_obs is None: 
    return None
  obs = lose_color(preprocess(obs))
  prev_obs = lose_color(preprocess(prev_obs))
  # obs_diff = obs - prev_obs
  # draw.draw(obs_diff, "obs.png")
  return obs, prev_obs

# generate a pong trace, the actor takes in the last 2 states as inputs
def generate_pong_trace(env, start_state, agent, n=200, do_render=True):
  env.reset()
  env.restore_full_state(start_state)

  trace = []
  all_obs = [None, None]

  for i in range(n):
    action = agent.act((all_obs[-2], all_obs[-1]))
    obs, reward, done, comments = env.step(action)
    if do_render:
      env.render()
    trace.append(((all_obs[-2], all_obs[-1]), action, reward))
    all_obs.append(obs)
    if done: break

  return trace

def get_random_state(env, start_state):
  env.reset()
  env.restore_full_state(start_state)
  for i in range(random.randint(100, 500)):
    _, a, b, c = env.step(random.choice([2,3]))
  state = env.clone_full_state()
  return state

