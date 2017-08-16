from env import *
from draw import *
from td_agent import *
from oracle_model import *
from stateless_regularized_model import *
import pickle
import sys

# set supervision to true or false
regularization = sys.argv[1]
assert regularization in ["yes", "no"], "regularization argument must be [yes] or [no]"
if regularization == "yes":
  regularization = True
else:
  regularization = False

L = 16
bugzero_train = BugZero(L)
test_mazes = pickle.load(open("test_datas/mazes.p","rb"))


# ================ TRAIN THE AGENT ================

def agent_xform(s):
  maze, start, end, path, info = s
  if len(path) == 0:
    last_step = np.array([0.0, 0.0])
  else:
    last = path[-1]
    last_pos, last_move = last
    last_step = np.array(bugzero_train.diff(start, last_pos))
    
  center_state = bugzero_train.centered_state(s)
  glimpse = bugzero_train.get_glimpse(center_state, 1)
  scent = bugzero_train.get_scent(center_state, 1)
# use this to remove scent
#  scent = np.zeros(shape=[3,3])
  goal_dir  = bugzero_train.get_goal_direction(center_state)
  start_dir = bugzero_train.get_start_direction(center_state)
  joint_state = np.concatenate([np.reshape(glimpse, [glimpse.size]), 
                                np.reshape(scent,   [scent.size]), 
                                goal_dir, start_dir])
  return joint_state

def agent_xform_action(a):
  ret = np.array([0.0, 0.0, 0.0, 0.0])
  ret[ bugzero_train.ACTIONS.index(a) ] = 1.0
  return ret

exp_size = 100
bug = StatelessAgentReg("bug", 22, 4, agent_xform, agent_xform_action, bugzero_train.ACTIONS)
experience = Experience(exp_size)

populate_experience(bugzero_train, bug, experience, exp_size)


for i in xrange(10000):
  maze = bugzero_train.gen_s()

  if regularization:
    transitionz = experience.n_sample_transition(500)
    st1, st2 = find_collapse_state(transitionz, agent_xform)
    bug.regularize(st1, st2)

  tr_agent = bugzero_train.get_trace(bug, maze)
  experience.add_balanced(tr_agent)

  batch = experience.n_sample(50)
  bug.learn(batch, "move")

  if i % 200 == 0:
    tr = bugzero_train.get_trace(bug, maze)
    path = bugzero_train.trace_to_path(tr)
    print "iteration ", i
    print path
    action_path = bugzero_train.trace_to_action_path(tr)
    draw(maze, "maze.png", path, action_path)

  if i % 2000 == 1999:
    print "accuracy ", get_accuracy_from_set(bug, bugzero_train, test_mazes)
    bug.save_model("models/bug_bug_model.ckpt")

