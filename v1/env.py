# a simple interpreter (environment)

# a STATE is a set

# we are only coding the problem of reversing a list now

import random
import copy

# ............................................................. generate an initial state
# it contains:
#              an array "ary"
#              two pointrs "pt1", "pt2"
#              program counter "pc"
def gen_s0(a_len = 5, a_val = 5, max_pc = 4, det=False):
  def get_rand_perm(a_len):
    ret = list(range(1,a_len+1))
    if not det:
      random.shuffle(ret)
    return ret
  the_ary = get_rand_perm(a_len)
  pc = [0 for i in range(max_pc)]
  pc[0] = 1
  pc = tuple(pc)
  return { "ary" : the_ary, 
           "pt1" : 0, 
           "pt2" : 0,
           "pc" : pc,
           "last" : "start"
         }

def correct(s0, sn):
  return s0["ary"] == list(reversed(sn["ary"]))

def stringify(state):
  arr = [state[k] for k in sorted(state.keys())]
  return str(arr)
    
# ............................................................................... actions
# actions are commands that can change the state to the next state

# move pointer 1 forward (safely, if already at end don't move)
def pt1_plus(s):
  ss = copy.deepcopy(s)
  ss["pt1"] = min(ss["pt1"] + 1, len(ss["ary"]) - 1)
  ss["last"] = "pt1_plus"
  return ss
  
# move pointer 2 forward (safely, if already at end don't move)
def pt2_plus(s):
  ss = copy.deepcopy(s)
  ss["pt2"] = min(ss["pt2"] + 1, len(ss["ary"]) - 1)
  ss["last"] = "pt2_plus"
  return ss
  
# move pointer 1 backward (safely, if already at end don't move)
def pt1_minu(s):
  ss = copy.deepcopy(s)
  ss["pt1"] = max(ss["pt1"] - 1, 0)
  ss["last"] = "pt1_minu"
  return ss
  
# move pointer 2 backward (safely, if already at end don't move)
def pt2_minu(s):
  ss = copy.deepcopy(s)
  ss["pt2"] = max(ss["pt2"] - 1, 0)
  ss["last"] = "pt2_minu"
  return ss

# swap the content at the two pointers
def swp(s):
  ss = copy.deepcopy(s)
  ary = ss["ary"]
  pt1_val = ary[ss["pt1"]]
  pt2_val = ary[ss["pt2"]]
  ary[ss["pt1"]] = pt2_val
  ary[ss["pt2"]] = pt1_val
  ss["last"] = "swp"
  return ss

# set the pc 
def set_pc1(s):
  ss = copy.deepcopy(s)
  ss["pc"] = (1,0,0,0)
  ss["last"] = "set_pc1"
  return ss

def set_pc2(s):
  ss = copy.deepcopy(s)
  ss["pc"] = (0,1,0,0)
  ss["last"] = "set_pc2"
  return ss

def set_pc3(s):
  ss = copy.deepcopy(s)
  ss["pc"] = (0,0,1,0)
  ss["last"] = "set_pc3"
  return ss

def set_pc4(s):
  ss = copy.deepcopy(s)
  ss["pc"] = (0,0,0,1)
  ss["last"] = "set_pc4"
  return ss

Actions = [
  "start",
  "pt1_plus",
  "pt2_plus",
  "pt1_minu",
  "pt2_minu",
  "swp",
  "set_pc1",
  "set_pc2",
  "set_pc3",
  "set_pc4",
  "end"
]

ActionsMap = {
  "start" : lambda x: x,
  "pt1_plus" : pt1_plus,
  "pt2_plus" : pt2_plus,
  "pt1_minu" : pt1_minu,
  "pt2_minu" : pt2_minu,
  "swp" : swp,
  "set_pc1" : set_pc1,
  "set_pc2" : set_pc2,
  "set_pc3" : set_pc3,
  "set_pc4" : set_pc4,
  "end" : lambda x: x
}

# print set(ActionsMap.keys())
# print set(Actions)
assert set(ActionsMap.keys()) == set(Actions)

# ............................................................................ predicates
# predicates help you abstract the concrete state

# pt1 > 0
def pt1_0(s):
  return int(s["pt1"] > 0)

# pt2 > 0
def pt2_0(s):
  return int(s["pt2"] > 0)

# pt1 < n
def pt1_n(s):
  return int(s["pt1"] < len(s["ary"]) - 1)

# pt2 < n
def pt2_n(s):
  return int(s["pt2"] < len(s["ary"]) - 1)

# pt1 >= pt2
def pt1_pt2(s):
  return int(s["pt1"] >= s["pt2"])

def abstract(s):
  preds = (pt1_0(s),
           pt2_0(s),
           pt1_n(s),
           pt2_n(s),
           pt1_pt2(s))
  last_moves = [0 for i in range(len(Actions))]
  last_moves[Actions.index(s["last"])] = 1
  return (preds, s["pc"], tuple(last_moves))

def abstract_old(s):
  preds = (pt1_0(s),
           pt2_0(s),
           pt1_n(s),
           pt2_n(s),
           pt1_pt2(s))
  last_moves = [0 for i in range(len(Actions))]
  last_moves[Actions.index(s["last"])] = 1
  return (preds, s["pc"], tuple(last_moves))

