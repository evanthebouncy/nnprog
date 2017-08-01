from env import *
from draw import *
from stateless_model import *
import random

L = 16
exp_size = 100

bugzero = BugZero(L)
bugzero_test = BugZero(2*L)

def xform(s, fake_call_state = None):
  maze, start, end, path, call_state = s
  if len(path) > 0:
    if path[-1][0] == start:
      call_state = [0.0, 1.0]
    if path[-1][0] != start:
      call_state = [1.0, 0.0]
#   if fake_call_state != None:
#     call_state = fake_call_state
  if len(path) == 0:
    last_step = np.array([0.0, 0.0])
  else:
    last = path[-1]
    last_pos, last_move = last
    last_step = np.array(bugzero.diff(start, last_pos))
    
  center_state = bugzero.centered_state(s)
  glimpse = bugzero.get_glimpse(center_state, 2)
  scent = bugzero.get_scent(center_state, 2)
# use this to remove scent
#  scent = np.zeros(shape=[3,3])
  goal_dir = bugzero.get_goal_direction(center_state)
  joint_state = np.concatenate([np.reshape(glimpse, [glimpse.size]), 
                                np.reshape(scent,   [scent.size]), 
                                goal_dir, np.array(call_state)])
  return joint_state

def xform_action(a):
  ret = np.array([0.0, 0.0, 0.0, 0.0])
  ret[ bugzero.ACTIONS.index(a) ] = 1.0
  return ret

# bug = StatelessAgent("bug", 22, 4, xform, xform_action, bugzero.ACTIONS)
bug = StatelessAgent("bug", 54, 4, xform, xform_action, bugzero.ACTIONS)
experience = Experience(exp_size)

while experience.check_success() < exp_size / 2:
  maze = bugzero.gen_s()
  tr = bugzero.get_trace(bug, maze, switch=False)
  experience.add_success(tr)
  print "succss trace cnt ",  experience.check_success()
for _ in range(exp_size / 2):
  maze = bugzero.gen_s()
  tr = bugzero.get_trace(bug, maze)
  experience.add(tr)

for i in xrange(1000000):
  maze = bugzero.gen_s()
  tr = bugzero.get_trace(bug, maze, switch=True)

  action_path = bugzero.trace_to_action_path(tr)
  experience.add_balanced(tr)

  batch = experience.n_sample(50)
  bug.learn(batch, "move")


  if i % 200 == 0:
    path = bugzero.trace_to_path(tr)
    print "iteration ", i
    print path
    action_path = bugzero.trace_to_action_path(tr)
    print action_path 
    draw(maze, "maze.png", path, action_path)

  if i % 200 == 100:
    maze = bugzero_test.gen_s()
    tr = bugzero_test.get_trace(bug, maze, switch=True)
    path = bugzero.trace_to_path(tr)
    print "iteration ", i
    print path
    action_path = bugzero.trace_to_action_path(tr)
    print action_path 
    draw(maze, "maze_test.png", path, action_path)

  if i % 2000 == 1999:
    print "accuracy ", get_accuracy(bug, bugzero_test, 1000)
    bug.save_model("models/bug_bug_model.ckpt")




