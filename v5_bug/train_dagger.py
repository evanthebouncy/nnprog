from env import *
from draw import *
from oracle_model import *
import random
from stateless_model import *

L = 16

bugzero = BugZero(L)
experience = Experience(10000)

def xform(state):
  vector_state = bugzero.vectorize_state(bugzero.centered_state(state))
  return vector_state

for i in range(20):
  maze = bugzero.gen_s()
  r_actor = bugzero.gen_a_star_actor(maze)
  trace = bugzero.get_trace(r_actor, maze)
  experience.add(trace)

def trace_to_path(trace):
  return [tr[0][1] for tr in trace]

def gen_batch(experience, n):

  input_states = []
  targets = []
  for _ in range(n):
    tr = experience.sample_transition()
    s,n_a,ss,r = tr
    name, a = n_a

    vect_state = bugzero.vectorize_state(bugzero.centered_state(s))
    action_num = bugzero.ACTIONS.index(a)

    targets.append(action_num)
    input_states.append(vect_state)

  return np.array(targets), np.array(input_states)

oracle = Oracle(L, xform, bugzero.ACTIONS)
oracle.restore_model("models/bug_oracle_model.ckpt")

print "accuracy ", get_accuracy(oracle, bugzero, 1000)
