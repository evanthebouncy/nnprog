from env import *

# print gen_s()
# 
# s0 = gen_s()
# 
# s1 = step(s0, "UP")
# s2 = step(s1, "LEFT")
# s3 = step(s2, "LEFT")
# 
# print s0, s1, s2, s3
# print abstract(s0), abstract(s1), abstract(s2), abstract(s3)
# 
# r_actor = RandomActor()
# for i in range(1000):
#   tr = get_trace(r_actor)
#   if tr[-1][-1] == 1.0:
#     print tr

flag = Flag()

r_actor = RandomActor(flag.ACTIONS)

for i in range(1000):
  tr = flag.get_trace(r_actor)
  if tr[-1][-1] == 1.0:
    print tr

