from env import *
from draw import *

L = 6

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

# flag = Flag()
# 
# r_actor = RandomActor(flag.ACTIONS)
# 
# for i in range(1000):
#   tr = flag.get_trace(r_actor)
#   if tr[-1][-1] == 1.0:
#     print tr
 
# bitdouble = BitDouble()
# 
# r_actor = RandomActor(bitdouble.ACTIONS)
# 
# for i in range(2000):
#   tr = bitdouble.get_trace(r_actor)
#   if tr[-1][-1] == 1.0:
#     print tr

bugzero = BugZero(L)
maze = bugzero.gen_s()
centered_maze = bugzero.centered_state(maze)

# path = bugzero.a_star_solution(maze)
draw(maze, "maze.png")
draw(centered_maze, "maze_center.png")

# # r_actor = RandomActor(bugzero.ACTIONS)
# 
# ctr = 0
# for i in range(2000):
#   print i, ctr
#   maze = bugzero.gen_s()
#   r_actor = bugzero.gen_a_star_actor(maze)
#   tr = bugzero.get_trace(r_actor, maze)
#   if tr[-1][-1] == 1.0:
#     ctr += 1
