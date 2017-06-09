from env import *
from util import *
import numpy as np
from collections import Counter

# actor critic learning
# critic: the usual TD-learning Q function critic
#         for each (s,a) pair, stores an explicit value for it
#         does NOT generate traces, only use traces to learn value
# actor: a stocastic actor with a soft-max policy
#        for each abs(s), store a vector of weights corresponding to actions
#        uses both traces and the Q values of the critic to learn

class Critic:
  def __init__(self, actions):
    self.learn_rate = 0.1
    self.discount = 0.9
    self.Q = dict()
    self.actions = actions

  # transform state from state to concret rep or abstr rep
  def state_xform(self, state):
    return stringify(state)

  def get_Q(self, s, a):
    s = self.state_xform(s)
    return self.Q[s,a]

  def get_action_score_from_s(self, state):
    scores = []
    for a in self.actions:
      s,a = state, a
      if (s,a) not in self.Q:
        self.Q[(s,a)] = 0.0
      scores.append((self.Q[(s,a)], a))
    return scores
  
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

class Actor:
  def __init__(self, actions):
    self.learn_rate = 0.1
    self.Q = dict()
    self.actions = actions
    self.det = False
    self.responsible = dict()

  # transform state from state to concret rep or abstr rep
  def state_xform(self, state):
    try:
      # return stringify(state)
      return abstract(state)
    except:
      return state

  def get_action_score_from_s(self, state):
    state = self.state_xform(state)
    scores = []
    for a in self.actions:
      s,a = state, a
      if (s,a) not in self.Q:
        self.Q[(s,a)] = 0.0
      scores.append((self.Q[(s,a)], a))
    return scores

  def get_move_prob(self, state):
    # transform the conc state to our represntation
    state = self.state_xform(state)
    scores_actions = self.get_action_score_from_s(state)
    # make sure there is a probability for every action always always
    scores = [x[0] for x in scores_actions] 
    actions = [x[1] for x in scores_actions]
    prob = softmax(scores)
    return actions, prob
  
  def get_move(self, state):
    if self.det:
      return self.get_move_det(state)
    actions, prob = self.get_move_prob(state)
    chosen_action = np.random.choice(actions, 1, p=prob)[0]
    
    # hack
#    if state["last"] == "swp":
#      return "pt2_minu"
#    if state["last"] == "pt2_minu":
#      return "pt1_plus"

    return chosen_action

  def get_move_det(self, state):
    actions, prob = self.get_move_prob(state)
    return sorted(zip(prob, actions))[-1][1]

#  def get_best_v(self, s):
#    scores = self.get_action_score_from_s(s)
#    return max(scores)[0]
    
  # trace is organized as a list of quadruples (s,a,s',r)
  def learn(self, trace, critic):
    for step in trace:
      s,a,ss,r = step
      # the state in trace needs to be transformed
      s_abs, ss_abs = self.state_xform(s), self.state_xform(ss)
      # make sure (s,a) is in the Q table
      if (s_abs,a) not in self.Q:
        self.Q[(s_abs,a)] = 0.0

      q_conc = critic.get_Q(s,a)
      actions, probs = self.get_move_prob(s)
      a_p_dict = dict(zip(actions, probs))
      # update delta for action entry
      deriv_a = q_conc * a_p_dict[a] * (1.0 - a_p_dict[a])
      self.Q[(s_abs,a)] += self.learn_rate * deriv_a      
      # update delta for non-action entry
      for b in a_p_dict:
        if b != a:
          deriv_b = q_conc * (-a_p_dict[b]) * a_p_dict[a]
          self.Q[(s_abs,b)] += self.learn_rate * deriv_b

      if q_conc > 0.0:
        if (s_abs,a) not in self.responsible:
          self.responsible[(s_abs,a)] = Counter()

        self.responsible[(s_abs,a)][(stringify(s),a)] += 1





