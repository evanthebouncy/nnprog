import gym
import gym.envs.atari
import random
import sys
from utils import *

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)

def get_future_projection(start_state):
  trejectory = []
  env_plan = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
  env_plan.reset()
  env_plan.restore_full_state(start_state)
  for i in range(100):
    obs, reward, done, info = env_plan.step(2)
    paddle = get_our_paddle(obs)
    ball = get_ball(obs)
    trejectory.append(ball)

  trejectory = [tr for tr in trejectory if tr != None]

  paddle_x = paddle[0]
  print "paddle x ", paddle_x
  trejectory = [(abs(ball_pos[0] - paddle_x), ball_pos[0] - paddle_x, ball_pos)\
                for ball_pos in trejectory]
  curr_dist = 9999
  for idx, tr in enumerate(trejectory):
    dist, ball_paddle_diff, ball_pos = tr
    if dist > curr_dist:
      _blah1, _blah2, prev_ball_pos = trejectory[idx-1]
      print "about to return something ", prev_ball_pos, ball_pos
      return prev_ball_pos[1]
    else:
      curr_dist = min(dist, curr_dist)

  print trejectory
  assert 0, "should never happen"

def act(state, ob_curr, ob_prev, prev_action):
#  obs, reward, done, info = env.step(0)

  prev_ball = get_ball(ob_prev) if ob_prev != None else None
  curr_ball = get_ball(ob_curr)
  
  # assert perfection 
#   if curr_ball != None:
#     assert curr_ball[0] < 75
  paddle = get_our_paddle(ob_curr)
  
  if (prev_ball == None or curr_ball == None or paddle == None) or\
      prev_ball[0] > curr_ball[0] or curr_ball[0] < 40: 
    target = 40
  else:
    future_ball_y = get_future_projection(state)
    target = future_ball_y
    print " target ", target

  if paddle[1] - target < 0:
    return 3
  else:
    return 2

# for state in states:
#   env.reset()
#   env.restore_full_state(state)
#   obs, _, _, _ = env.step(0)
#   done = False
#   while not done:
#     action = act(env.clone_full_state(), obs)
#     print action
#     obs, a, done, c = env.step(action)
#     env.render()
  
to_store = []

env.reset()
state = env.clone_full_state()
done = False
prev_obs = None
obs, a, done, c = env.step(2)
actionz = [2]
prev_action = 2

ctr = 0
while not done:
  ctr += 1
  print ctr
  # if ctr == 1000: break
  action = act(env.clone_full_state(), obs, prev_obs, prev_action)
  prev_action = action
  prev_obs = obs
  obs, reward, done, c = env.step(action)
  env.render()
  actionz.append(action)

to_store.append((state, 0.0, actionz))

import pickle
pickle.dump( to_store, open( "datas/train_pong.p", "wb" ) )
