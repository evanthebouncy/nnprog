import gym
import gym.envs.atari
import random
import sys
from utils import *

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
start_state = env.clone_full_state()

planner_traces = pickle.load(open("datas/planner_traces.p","rb"))

class RecordAgent:
  def __init__(self, recorded_actions):
    self.actions = recorded_actions
    self.counter = 0

  # takes some observation but doesn't really use them
  def act(self, s, show_prob):
    obs_prev, obs, prev_a = s
    assert self.counter < len(self.actions), "ran out of actions!!"
    ret = self.actions[self.counter]
    self.counter += 1
    return ret
    
def sample_planner_sa(n_pairs = 40):
  state, r_best, best_as = random.choice(planner_traces)
  rec_agent = RecordAgent(best_as)
  trace = generate_pong_trace(env, state, rec_agent, do_render=False)
  trs = [random.choice(trace) for i in range(n_pairs)]
  return [(pz[0], pz[1]) for pz in trs]

