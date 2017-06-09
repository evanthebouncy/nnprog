import numpy as np

L = 6
ACTIONS = ["UP", "LEFT", "RIGHT"]

def gen_s():
  ret = np.random.randint(-L,L,size=2)
  ret[1] = L
  if goal(ret): return gen_s()
  else: return ret
  

def step(s, a):
  if a == "UP": return s - np.array([1,2]) 
  if a == "LEFT": return s - np.array([-2,-1]) 
  if a == "RIGHT": return s - np.array([2,-1]) 
  else: assert 0

def goal(s):
  return s[0] == 0 and s[1] == 0


def get_trace(actor, bound=10, s=None):
  trace = []
  s = gen_s() if s == None else s
  move_reward = -0.1
  for i in range(bound):
    action = actor.act(s)
    ss = step(s, action)
    reward = 1.0 if goal(ss) else move_reward
    trace.append((s,action,ss,reward))
    if goal(ss): return trace
    s = ss
  return trace

def abstract(s):
  diffx, diffy = s[0], s[1]
  if diffy > 0: return "U"
  if diffx <= 0: return "L"
  else: return "R"

  return vertical + horizontal

class RandomActor:
  def act(self, s):
    ret = ACTIONS[np.random.randint(3)]
    return ret

def print_Q(Q):
  keys = sorted(list(Q.keys()))
  print keys
  for k in keys:
    print k, Q[eval(k)]

