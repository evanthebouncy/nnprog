from env import *
from draw import *
from stateless_model import *
import random

L = 16
exp_size = 100

bugzero = BugZero(L)

def xform(s, fake_call_state = None):
  maze, start, end, path, call_state = s
  if fake_call_state != None:
    call_state = fake_call_state
  if len(path) == 0:
    last_step = np.array([0.0, 0.0])
  else:
    last = path[-1]
    last_pos, last_move = last
    last_step = np.array(bugzero.diff(start, last_pos))
    
  center_state = bugzero.centered_state(s)
  glimpse = bugzero.get_glimpse(center_state, 1)
  scent = bugzero.get_scent(center_state, 1)
# use this to remove scent
#  scent = np.zeros(shape=[3,3])
  goal_dir = bugzero.get_goal_direction(center_state)
  joint_state = np.concatenate([np.reshape(glimpse, [glimpse.size]), 
                                np.reshape(scent,   [scent.size]), goal_dir, np.array(call_state)])
  return joint_state

def xform_action(a):
  ret = np.array([0.0, 0.0, 0.0, 0.0])
  ret[ bugzero.ACTIONS.index(a) ] = 1.0
  return ret

bug = StatelessAgent("bug", 22, 4, xform, xform_action, bugzero.ACTIONS)
experience = Experience(exp_size)
experience_pain = Experience(exp_size)

while experience.check_success() < exp_size / 2:
  maze = bugzero.gen_s()
  tr = bugzero.get_trace(bug, maze, switch=False)
  experience.add_success(tr)
  experience_pain.add_success(tr)
  print "succss trace cnt ",  experience.check_success()
for _ in range(exp_size / 2):
  maze = bugzero.gen_s()
  tr = bugzero.get_trace(bug, maze)
  experience.add(tr)
  experience_pain.add(tr)

blue_ok_pink_ok = 0
blue_ok_pink_bad = 0
blue_bad_pink_ok = 0
blue_bad_pink_bad = 0
all_blue = 0

for i in xrange(1000000):
  maze = bugzero.gen_s()
  tr = bugzero.get_trace(bug, maze, switch=True)
  pain_tr = bugzero.get_trace(bug, maze, switch=False)

  action_path = bugzero.trace_to_action_path(tr)
  # there is a hand off to megenta
  if 'meg' in str(action_path):
    # print tr[-1]
    # print pain_tr[-1]
    if bugzero.goal(tr[-1][-2]) and bugzero.goal(pain_tr[-1][-2]):
      blue_ok_pink_ok += 1
    if bugzero.goal(tr[-1][-2]) and not bugzero.goal(pain_tr[-1][-2]):
      blue_bad_pink_ok += 1
      experience.add_balanced(tr)
    if not bugzero.goal(tr[-1][-2]) and bugzero.goal(pain_tr[-1][-2]):
      blue_ok_pink_bad += 1
    if not bugzero.goal(tr[-1][-2]) and not bugzero.goal(pain_tr[-1][-2]):
      blue_bad_pink_bad += 1
      experience.add_balanced(tr)

  # if there is no hand-off to megenta, always add learning
  if 'meg' not in str(action_path):
    experience.add_balanced(tr)
    all_blue += 1
  
  experience_pain.add_balanced(pain_tr)

  batch = experience.n_sample(50)
  bug.learn(batch, "move")

  pain_batch = experience_pain.n_sample(50)
  bug.learn(pain_batch, "pain")

  if i % 200 == 0:
    path = bugzero.trace_to_path(tr)
    pain_path = bugzero.trace_to_path(pain_tr)
    print "iteration ", i
    print path
    action_path = bugzero.trace_to_action_path(tr)
    pain_action_path = bugzero.trace_to_action_path(pain_tr)
    print action_path 
    draw(maze, "maze.png", path, action_path)
    draw(maze, "maze_pain.png", pain_path, pain_action_path)
    print "whatev mishandle beautiful hopeless"
    print blue_ok_pink_ok, blue_ok_pink_bad, blue_bad_pink_ok, blue_bad_pink_bad, all_blue

  if i % 2000 == 1999:
    print "accuracy ", get_accuracy(bug, bugzero, 1000)
    bug.save_model("models/bug_bug_model.ckpt")

print "finished training, measuring accuracy "
print "accuracy ", get_accuracy(bug, bugzero, 1000)



