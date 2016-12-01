from random_agent import *
from run import *

random_policy = gen_random_policy()
 
s = gen_s0(5,5,4)
sn = run_star(s, random_policy, ActionsMap)

