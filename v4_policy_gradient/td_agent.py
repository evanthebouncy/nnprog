from env import *
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

  def __init__(self, env, name="td"):
    self.name = name
    self.env = env
    self.learn_rate = 0.1
    self.explore_rate = 0.1
    self.discount = 0.9
    self.Q = Dict()
    self.actions = env.ACTIONS

  def get_action_score_from_s(self, state):
    scores = []
    for a in self.actions:
      s,a = state, a
      scores.append((self.Q[(s,a)], a))
    return scores
  
  def act(self, state, env):
    scores = self.get_action_score_from_s(state)
    if random.random() < self.explore_rate:  
      return self.name, random.choice(self.actions)
    else:
      return self.name, max(scores)[1]

  def get_best_v(self, s):
    scores = self.get_action_score_from_s(s)
    return max(scores)[0]
    
  # trace is organized as a list of quadruples (s,a,s',r)
  def learn(self, traces, env):
    for trace in traces:
      for step in trace:
        s,name_a,ss,r = step
        name, a = name_a
        best_future_v = self.get_best_v(ss)
        td = (r + self.discount * best_future_v) - self.Q[(s,a)]
        self.Q[(s,a)] = self.learn_rate * td + self.Q[(s,a)]

def contraction(trace, agent_name):
  ret = []
  
  cur_s = None
  cur_a = None
  cur_ss = None
  cur_r = None
  for s, name_a, ss, r in trace: 
    name, a = name_a
    if name == agent_name and type(cur_s) != type(None):
      cur_ss = s
      ret.append((cur_s, (name, cur_a), cur_ss, cur_r))
      cur_s = s
      cur_a = a
      cur_ss = None
      cur_r = r
    if name == agent_name and type(cur_s) == type(None):
      cur_s = s
      cur_a = a
      cur_ss = None
      cur_r = r
    if name != agent_name and type(cur_r) != type(None):
      cur_r += r
      cur_ss = s

  last_s, last_a, last_ss, last_r = trace[-1]
  ret.append((cur_s, (name, cur_a), last_ss, cur_r))

  if cur_a == None: return []

  return ret
      

class TandemAgent:

  def __init__(self, agent_oracle, agent_student, split_cond):
    self.agent_oracle = agent_oracle
    self.agent_student = agent_student
    self.prob = 0.5
    self.abst = []

    assert split_cond in ["prob", "abst"]
    self.split_cond = split_cond

  def set_prob(self, p):
    self.prob = p

  def set_abst(self, abst):
    self.abst = abst

  def split(self, state, env):
    if self.split_cond == "abst": return env.abstract(state) in self.abst
    if self.split_cond == "prob": return np.random.random() > self.prob

  def act(self, state, env):
    if self.split(state, env):
      return self.agent_student.act(state, env)
    else:
      return self.agent_oracle.act(state, env)

  def learn(self, traces, env):
    name_oracle = self.agent_oracle.name
    name_student = self.agent_student.name

    player_oracle_traces, player_student_traces = [], []
    for trace in traces:
      trace_oracle = contraction(trace, name_oracle)
      player_oracle_traces.append(trace_oracle)
      trace_student = contraction(trace, name_student)
      player_student_traces.append(trace_student)

    # print "learning "
    # print traces[0]
    # print player_oracle_traces[0]
    # print player_student_traces[0]
    self.agent_oracle.learn(player_oracle_traces, env)
    self.agent_student.learn(player_student_traces, env)




      




