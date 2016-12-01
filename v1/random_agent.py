from env import *
import random

def gen_random_policy():
  
  policy_mapping = dict()

  def random_policy(s):
    # abs_s = abstract(s)
    abs_s = str(s)
    if abs_s not in policy_mapping:
      policy_mapping[abs_s] = random.choice(Actions) 
    print policy_mapping
    return policy_mapping[abs_s]

  return random_policy

