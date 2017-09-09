import gym
import gym.envs.atari
import random
import sys
from utils import *

cache = dict()

num_states = 1

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)

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

def get_future_projection(start_state):
  trejectory = []
  env_plan = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
  env_plan.reset()
  env_plan.restore_full_state(start_state)
  for i in range(100):
    obs, reward, done, info = env_plan.step(0)
    paddle = get_our_paddle(obs)
    ball = get_ball(obs)
    trejectory.append(ball)

  trejectory = [tr for tr in trejectory if tr != None]

  paddle_x = paddle[0]
  trejectory = [(abs(ball_pos[0] - paddle_x), ball_pos) for ball_pos in trejectory]
  curr_dist = 9999
  for tr in trejectory:
    dist, ball_pos = tr
    if dist > curr_dist:
      return ball_pos[1]
    else:
      curr_dist = min(dist, curr_dist)

  print trejectory
  assert 0, "should never happen"

def act(state, ob_prev):
  obs, reward, done, info = env.step(0)

  prev_ball = get_ball(ob_prev)
  curr_ball = get_ball(obs)
  paddle = get_our_paddle(ob_prev)
  if (prev_ball == None or curr_ball == None) or\
      prev_ball[0] > curr_ball[0] or curr_ball[0] < 40: 
    target = 40
  else:
    future_ball_y = get_future_projection(state)
    print paddle[1], future_ball_y
    target = future_ball_y

#  if abs(paddle[1] - target) > 5:
  if paddle[1] - target < 0:
    return 3
  else:
    return 2
#  else:
#    return 0

for state in states:
  env.reset()
  env.restore_full_state(state)
  obs, _, _, _ = env.step(0)
  done = False
  while not done:
    action = act(env.clone_full_state(), obs)
    print action
    obs, a, done, c = env.step(action)
    env.render()
  
# import pickle
# pickle.dump( to_store, open( "datas/train_pong.p", "wb" ) )


