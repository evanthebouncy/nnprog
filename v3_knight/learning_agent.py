from env import *
import learning_agent
import random
from numpy import array

def argmax(iterable):
  return max(enumerate(iterable), key=lambda x: x[1])[0]

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

  def __init__(self, env, use_abs):
    self.env = env
    self.learn_rate = 0.1
    self.explore_rate = 0.1
    self.discount = 0.9
    self.Q = Dict()
    self.actions = env.ACTIONS
    self.use_abs = use_abs

  def xform(self, s):
    if type(s) == type(""): return s
    if self.use_abs: return self.env.abstract(s)
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

  def learn_with_critic(self, trace, critic):
    for step in trace:
      s,a,ss,r = step
      ss_orig = ss
      s, ss = self.xform(s), self.xform(ss)

      # use the critic for the value for concrete ss
      best_future_v = critic.get_best_v(ss_orig)

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
  
class AngelicLearn(TDLearn):
  def __init__(self, env):
    TDLearn.__init__(self, env, True)
    self.to_abs = dict()

  def xform(self, s):
    if type(s) == type(""): return s
    if self.env.abstract(s) in self.to_abs: return self.env.abstract(s)
    return s

  def get_best_v(self, s):
    usual_v = TDLearn.get_best_v(self, s)
    if self.xform(s) in list(self.to_abs.keys()):
      # print s, self.xform(s), 
      ret = self.Q[(self.xform(s), self.to_abs[self.xform(s)])]
      # print ret
      return ret
    else:
      return usual_v

  def act(self, state):
    usual_act = TDLearn.act(self, state)
    if self.env.abstract(state) in self.to_abs:
      return self.to_abs[self.env.abstract(state)]
    else:
      return usual_act

def synthesize(env, angl_mkr, train_iter, test_iter):
  abs_states, actions = env.STATES, env.ACTIONS
  policy = dict()
  for a_state in abs_states:
  
    attempts = []
    for action in actions:
      attempt = dict()
      for k in policy:
        attempt[k] = policy[k]
      attempt[a_state] = action
      attempts.append(attempt)

    agent_angls = []
    for aaa in attempts:
      agent_angl = angl_mkr(env)
      agent_angl.to_abs = aaa
      agent_angls.append(agent_angl)

    for i in range(train_iter):
      s = env.gen_s()

      for aa in agent_angls:
        tr_angl = env.get_trace(aa, s=s)
        aa.learn(tr_angl)

    for aa in agent_angls:
      aa.explore_rate = 0.0

    counts = [0 for _ in range(len(agent_angls))]
    for i in range(test_iter):
      s = env.gen_s()
      for idd, aa in enumerate(agent_angls):
        tr_angl = env.get_trace(aa, s=s)

        if tr_angl[-1][-1] == 1.0:
          counts[idd] += 1

    best_attempt = attempts[argmax(counts)]
    print best_attempt, max(counts), zip(attempts, counts)
    policy = best_attempt

  return policy 
