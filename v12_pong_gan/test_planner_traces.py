import pickle
import gym
import gym.envs.atari
import random
from gans import RecordAgent
from utils import *

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
planner_traces = pickle.load(open("datas/planner_traces.p","rb"))

state, r_best, best_as = random.choice(planner_traces)
rec_agent = RecordAgent(best_as)
print generate_pong_trace(env, state, rec_agent)

