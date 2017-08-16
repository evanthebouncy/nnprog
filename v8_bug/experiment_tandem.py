from env import *
from draw import *
from td_agent import *
from oracle_model import *
from stateless_model import *
import sys

L = 16
bugzero_train = BugZero(L)
bugzero_test = BugZero(2*L)

# =============== LOAD IN THE ORACLE ===============
def oracle_xform(state):
  vector_state = bugzero_train.vectorize_state(bugzero_train.centered_state(state))
  return vector_state

oracle = Oracle(L, oracle_xform, bugzero_train.ACTIONS)
oracle.restore_model("models/bug_oracle_model.ckpt")
print "oracle accuracy ", get_accuracy(oracle, bugzero_train, 100)


# ================ TRAIN THE AGENT ================

def agent_xform(s, fake_call_state = None):
  maze, start, end, path, call_state = s
    
  center_state = bugzero_train.centered_state(s)
  glimpse = bugzero_train.get_glimpse(center_state, 1)
  scent = bugzero_train.get_scent(center_state, 1)
# use this to remove scent
#  scent = np.zeros(shape=[3,3])
  goal_dir = bugzero_train.get_goal_direction(center_state)
  joint_state = np.concatenate([np.reshape(glimpse, [glimpse.size]), 
                                np.reshape(scent,   [scent.size]), goal_dir, np.array(call_state)])
  return joint_state

def agent_xform_action(a):
  ret = np.array([0.0, 0.0, 0.0, 0.0])
  ret[ bugzero_train.ACTIONS.index(a) ] = 1.0
  return ret

exp_size = 100
bug = StatelessAgent("bug", 22, 4, agent_xform, agent_xform_action, bugzero_train.ACTIONS)
experience = Experience(exp_size)

tandem_agent = TandemAgent(oracle, bug)

# print "populating experience "
# if supervision:
#   populate_experience(bugzero_train, oracle, experience, exp_size)
# else: 
#   populate_experience(bugzero_train, bug, experience, exp_size)
# 
# 
# for i in xrange(10000):
#   maze = bugzero_train.gen_s()
# 
#   if supervision:
#     tr_oracle = bugzero_train.get_trace(oracle, maze)
#     experience.add(tr_oracle)
#   else:
#     tr_agent = bugzero_train.get_trace(bug, maze)
#     experience.add_balanced(tr_agent)
# 
#   batch = experience.n_sample(50)
#   bug.learn(batch, "move")
# 
#   if i % 200 == 0:
#     tr = bugzero_train.get_trace(bug, maze)
#     path = bugzero_train.trace_to_path(tr)
#     print "iteration ", i
#     print path
#     action_path = bugzero_train.trace_to_action_path(tr)
#     draw(maze, "maze.png", path, action_path)
# 
#   if i % 2000 == 1999:
#     print "accuracy ", get_accuracy(bug, bugzero_test, 1000)
#     bug.save_model("models/bug_bug_model.ckpt")
# 
