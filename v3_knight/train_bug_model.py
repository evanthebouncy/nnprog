from env import *
from draw import *
from bug_model import *
import random

L = 16

bugzero = BugZero(L)
experience = Experience(10000)

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
    tr = experience.sample()
    s,a,ss,r = tr

    vect_state = bugzero.vectorize_state(bugzero.centered_state(s))
    action_num = bugzero.ACTIONS.index(a)

    targets.append(action_num)
    input_states.append(vect_state)

  return np.array(targets), np.array(input_states)

oracle = Oracle(L)
# oracle.restore_model("models/bug_oracle_model.ckpt")



for i in range(100000):

  maze = bugzero.gen_s()
  r_actor = bugzero.gen_a_star_actor(maze)
  trace = bugzero.get_trace(r_actor, maze)
  experience.add(trace)

  batch = gen_batch(experience, 50)
  oracle.train_model(batch)

  if i % 200 == 0:
    oracle.save_model("models/bug_oracle_model.ckpt")

    oracle.act(maze, env=bugzero)
    trace_oracle = bugzero.get_trace(oracle, maze)
    path = trace_to_path(trace_oracle)
    print path
    draw(maze, "maze.png", path)



