from env import *
from draw import *
from stateless_model import *
import random
from human_agent import *

L = 16
exp_size = 100

bugzero = BugZero(L)
bugzero_test = BugZero(2*L)

bug = Human1(bugzero)

maze = bugzero_test.gen_s()
tr = bugzero_test.get_trace(bug, maze)
path = bugzero.trace_to_path(tr)
print path
draw(maze, "maze_human.png", path)

print "accuracy ", get_accuracy(bug, bugzero_test, 1000)


