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
  for i in range(10):
    obs, reward, done, info = env_plan.step(2)
    paddle = get_our_paddle(obs)
    ball = get_ball(obs)
    trejectory.append(ball)

  trejectory = [tr for tr in trejectory if tr != None]

  paddle_x = paddle[0]
  print "paddle x ", paddle_x
  trejectory = [(abs(ball_pos[0] - paddle_x), ball_pos[0] - paddle_x, ball_pos)\
                for ball_pos in trejectory]

  all_ball_dist_pos = []
  curr_dist = 9999
  for idx, tr in enumerate(trejectory):
    dist, ball_paddle_diff, ball_pos = tr
    all_ball_dist_pos.append((dist, ball_pos))
    if dist > curr_dist:
      _blah1, _blah2, prev_ball_pos = trejectory[idx-1]
      print "about to return something ", prev_ball_pos, ball_pos
      return prev_ball_pos[1]
    else:
      curr_dist = min(dist, curr_dist)

  # debugging 
  # print trejectory
  # # assert 0, "should never happen"
  # print " .................................................. shouldnt be here . . ."
  if all_ball_dist_pos == []: return paddle[1]
  print all_ball_dist_pos
  return min(all_ball_dist_pos, key=lambda x: x[0])[1][1]

def act(state, ob_curr, ob_prev, prev_action):
#  obs, reward, done, info = env.step(0)

  prev_ball = get_ball(ob_prev) if ob_prev != None else None
  curr_ball = get_ball(ob_curr)
  
  # assert perfection 
#   if curr_ball != None:
#     assert curr_ball[0] < 75
  paddle = get_our_paddle(ob_curr)
  
  if (prev_ball is None or curr_ball is None or paddle is None) or\
      prev_ball[0] > curr_ball[0]:
    target = paddle[1]
  else:
    future_ball_y = get_future_projection(state)
    target = future_ball_y
    print " target ", target

  def brake(prev_action):
    return 5 - prev_action

  if paddle[1] - target < -2:
    return 3
  if paddle[1] - target > 2:
    return 2
  return brake(prev_action)

to_store = []
start_state = env.clone_full_state()

for i in range(200):
  state = get_random_state(env, start_state)
  env.reset()
  env.restore_full_state(state)
  done = False
  prev_obs = None
  obs, a, done, c = env.step(2)
  actionz = [2]
  prev_action = 2

  ctr = 0
  while not done and ctr < 202:
    ctr += 1
    print "game number ", i, "iteration ", ctr
    # if ctr == 1000: break
    action = act(env.clone_full_state(), obs, prev_obs, prev_action)
    prev_action = action
    prev_obs = obs
    obs, reward, done, c = env.step(action)
    env.render()
    actionz.append(action)

  to_store.append((state, 0.0, actionz))

  import pickle
  print "storing "
  pickle.dump( to_store, open( "datas/planner_traces.p", "wb" ) )
