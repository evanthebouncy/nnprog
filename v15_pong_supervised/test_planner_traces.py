import pickle
import gym
import gym.envs.atari
import random
from supervise import RecordAgent
from utils import *

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
planner_traces = pickle.load(open("datas/planner_traces.p","rb"))
print "num traces "
print len(planner_traces)
print "length per trace "
print len(planner_traces[0][2])

while True:
  state, r_best, best_as = random.choice(planner_traces)
  rec_agent = RecordAgent(best_as)
  trr = generate_pong_trace(env, state, rec_agent)
  # print trr

