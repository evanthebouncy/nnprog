from human_agent import *
from run import *

s = gen_s0(1,5,4)
sn = run_star(s, human_policy, ActionsMap)
assert correct(s, sn)
s = gen_s0(2,5,4)
sn = run_star(s, human_policy, ActionsMap)
assert correct(s, sn)
s = gen_s0(3,5,4)
sn = run_star(s, human_policy, ActionsMap)
assert correct(s, sn)
s = gen_s0(4,5,4)
sn = run_star(s, human_policy, ActionsMap)
assert correct(s, sn)
s = gen_s0(5,5,4)
sn = run_star(s, human_policy, ActionsMap)
assert correct(s, sn)
s = gen_s0(6,5,4)
sn = run_star(s, human_policy, ActionsMap)
assert correct(s, sn)

