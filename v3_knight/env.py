import numpy as np
from numpy import array
import copy

class Knight:
  L = 6
  STATES = ["U", "L", "R"]
  ACTIONS = ["UP", "LEFT", "RIGHT"]

  def gen_s(self):
    L = self.L
    ret = np.random.randint(-L,L,size=2)
    # ret[1] = L
    if self.goal(ret): return self.gen_s()
    else: return ret
    

  def step(self, s, a):
    if a == "UP": return s - np.array([1,2]) 
    if a == "LEFT": return s - np.array([-2,-1]) 
    if a == "RIGHT": return s - np.array([2,-1]) 
    else: assert 0

  def goal(self, s):
    return s[0] == 0 and s[1] == 0


  def get_trace(self, actor, bound=10, s=None):
    trace = []
    s = gen_s() if s == None else s
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s)
      ss = self.step(s, action)
      reward = 1.0 if self.goal(ss) else move_reward
      trace.append((s,action,ss,reward))
      if self.goal(ss): return trace
      s = ss
    return trace

  def abstract(self, s):
    diffx, diffy = s[0], s[1]
    if diffy > 0: return "U"
    if diffx <= 0: return "L"
    else: return "R"

    return vertical + horizontal

class Flag:
  L = 5
  STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]
  ACTIONS = ["1++", "2--", "swp", "stop"]

  def gen_s(self):
    L = self.L
    pt1, pt2 = 0, L-1
    ret = pt1, pt2, np.random.randint(0,2,size=L)
    if self.goal(ret): return self.gen_s()
    else: return ret

  def goal(self, state):
    _,_,s = state
    flag = False
    for x in s:
      if x == 1:
        flag = True
      if x == 0 and flag:
        return False
    return True 

  def step(self, s, a):
    p1, p2, ary = s
    if a == "1++": return min(p1 + 1, self.L - 1), p2, ary
    if a == "2--": return p1, max(p2 - 1, 0), ary
    if a == "swp": 
      new_ary = np.copy(ary)
      new_ary[p1] = ary[p2]
      new_ary[p2] = ary[p1]
      return p1, p2, new_ary
    if a == "stop":
      return s
    else: assert 0

  def get_trace(self, actor, bound=12, s=None):
    trace = []
    s = self.gen_s() if s == None else s
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s)
      ss = self.step(s, action)

      reward = 1.0 if (self.goal(ss) and action == "stop") else move_reward
      trace.append((s,action,ss,reward))
      if action == "stop": return trace
      s = ss
    return trace

  def abstract(self, s):
    p1, p2, ary = s
    ret = ""
    ret += "1" if p1 == p2 else "0"
    ret += "1" if ary[p1] == 1 else "0"
    ret += "1" if ary[p2] == 1 else "0"
    return ret

# doubling the number in bit representation
class BitDouble:
  L = 5
  # the four pointers are ptr1, carry, output
  # the values of pointers can be 0 or 1
  # pt1 and output are pointing at index 0 in beginning while carry points at index 1
  # all pointers increments by 1 with the ++ command
  # there is a program counter which always increases but can be reset
  STATES = []
  ACTIONS = ["++", "c0", "c1", "o0", "o1"]

  def gen_s(self):
    L = self.L
    pt1, ptc, pto, PC = 0, 1, 0, 0
    a1 = list(np.random.randint(0,2,size=L-1)) + [0]
    ao = [0 for i in range(L)]
    return pt1, ptc, pto, a1, ao, PC

  def __init__(self):
    for p1_content in ["0", "1"]:
      for c_content in ["0", "1"]:
        for o_content in ["0", "1"]:
          for pc in ["0", "1"]:
            self.STATES.append(p1_content+c_content+o_content+pc)


  def goal(self, state):
    pt1, ptc, pto, a1, ao, PC = state
    def meow(aaa):
      ret = ""
      for a in aaa:
        ret = str(a) + ret
      return int(ret, 2)
    return meow(a1) + meow(a1) == meow(ao)

  def step(self, s, a):
    pt1, ptc, pto, a1, ao, PC = copy.deepcopy(s)
    if a == "++": return min(pt1 + 1, self.L - 1),\
                         min(ptc + 1, self.L - 1),\
                         min(pto + 1, self.L - 1), a1, ao, (PC+1) % 2
    if a == "c0": 
      ao[ptc] = 0
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
    if a == "c1": 
      ao[ptc] = 1
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
    if a == "o0": 
      ao[pto] = 0
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
    if a == "o1": 
      ao[pto] = 1
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
 #   if a == "reset":
 #     return pt1, ptc, pto, a1, ao, 0
    assert 0

  def get_trace(self, actor, s=None):
    bound = self.L * 6
    def stop(s):
      pt1, ptc, pto, a1, ao, PC = copy.deepcopy(s)
      return ptc == pto

    trace = []
    s = self.gen_s() if s == None else s
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s)
      ss = self.step(s, action)

      reward = 1.0 if (stop(ss) and self.goal(ss)) else move_reward
      trace.append((s,action,ss,reward))
      if stop(ss): return trace
      s = ss
    return trace

  def abstract(self, s):
    pt1, ptc, pto, a1, ao, PC = copy.deepcopy(s)
    ret = ""
    ret += str(a1[pt1])
    ret += str(ao[ptc])
    ret += str(ao[pto])
    ret += str(PC)
    return ret

class RandomActor:
  def __init__(self, actions):
    self.actions = actions

  def act(self, s):
    ret = self.actions[np.random.randint(len(self.actions))]
    return ret

def print_Q(Q):
  keys = sorted(list(Q.keys()))
  print keys
  for k in keys:
    print k, Q[eval(k)]

