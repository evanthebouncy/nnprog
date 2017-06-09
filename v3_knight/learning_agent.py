from env import *
import learning_agent
import random
from numpy import array

class Dict(dict):
  def __setitem__(self, key, value):
    key = repr(key)
    super(Dict, self).__setitem__(key, value)

  def __getitem__(self, key):
    key = repr(key)
    try:
      return super(Dict, self).__getitem__(key)
    except:
      return 0.0


class TDLearn:

  def __init__(self, actions, use_abs):
    self.learn_rate = 0.1
    self.explore_rate = 0.1
    self.discount = 0.9
    self.Q = Dict()
    self.actions = actions
    self.use_abs = use_abs

  def xform(self, s):
    if type(s) == type(""): return s
    if self.use_abs: return abstract(s)
    return s

  def get_action_score_from_s(self, state):
    state = self.xform(state)
    scores = []
    for a in self.actions:
      s,a = state, a
      scores.append((self.Q[(s,a)], a))
    return scores
  
  def act(self, state):
    state = self.xform(state)
    scores = self.get_action_score_from_s(state)
    if random.random() < self.explore_rate:  
      return random.choice(self.actions)
    else:
      return max(scores)[1]

  def get_best_v(self, s):
    scores = self.get_action_score_from_s(s)
    return max(scores)[0]
    
  # trace is organized as a list of quadruples (s,a,s',r)
  def learn(self, trace):
    for step in trace:
      s,a,ss,r = step
      s, ss = self.xform(s), self.xform(ss)
      best_future_v = self.get_best_v(ss)
      td = (r + self.discount * best_future_v) - self.Q[(s,a)]
      self.Q[(s,a)] = self.learn_rate * td + self.Q[(s,a)]

# take a concrete agent and refine it by average
def avg_refine(agent_conc, abstract):
  newQ = Dict()
  newQ_cnt = Dict()
  concQ = agent_conc.Q
  for k in concQ.keys():
    s,a = eval(k)
    abs_s = abstract(s)

    all_scores = [concQ[(s,aa)] for aa in ACTIONS]
    # print all_scores
    newQ[(abs_s,a)] += 1.0 if concQ[(s,a)] == max(all_scores) and concQ[(s,a)] != 0 else 0.0
    newQ_cnt[(abs_s,a)] += 1

  ctr_abst = learning_agent.TDLearn(ACTIONS, True)

  newQQ = Dict()
  keys = newQ.keys()
  for k in keys:
    newQQ[eval(k)] = newQ[eval(k)] / newQ_cnt[eval(k)]

  ctr_abst.Q = newQQ

  return ctr_abst
  
