from env import *
import random

class TDLearn:

  def __init__(self, actions, use_conc):
    self.learn_rate = 0.1
    self.explore_rate = 0.1
    self.discount = 0.9
    self.Q = dict()
    self.actions = actions
    # do learning / planning over concrete state or no
    self.use_conc = use_conc

  # transform state from state to concret rep or abstr rep
  def state_xform(self, state):
    if self.use_conc:
      return stringify(state)
    else:
      return abstract(state)

  def get_action_score_from_s(self, state):
    scores = []
    for a in self.actions:
      s,a = state, a
      if (s,a) not in self.Q:
        self.Q[(s,a)] = 0.0
      scores.append((self.Q[(s,a)], a))
    return scores
  
  def get_move(self, state):
    # transform the conc state to our represntation
    state = self.state_xform(state)
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
      # the state in trace needs to be transformed
      s, ss = self.state_xform(s), self.state_xform(ss)
      # make sure (s,a) is in the Q table
      if (s,a) not in self.Q:
        self.Q[(s,a)] = 0.0

      best_future_v = self.get_best_v(ss)
      td = (r + self.discount * best_future_v) - self.Q[(s,a)]
      # update
      self.Q[(s,a)] += self.learn_rate * td

