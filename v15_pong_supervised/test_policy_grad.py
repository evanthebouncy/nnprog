from supervise import *
from stateless_model import *
from discriminator import *
from draw import *

def xform_full(state):
  ob1, ob2, prev_a = state
  prev_a = xform_action(prev_a)
#  return get_simple_signal(ob1, ob2)
  if ob1 is None or ob2 is None:
    return np.zeros([6400*2+2])
  else:
    ob1, ob2 = get_signal_full_image(ob1, ob2)
    diff = ob2 - ob1
    together = np.concatenate([diff, ob2], axis=1)
    #draw(together, "together.png")
    #assert 0
    
    together_flat = np.reshape(together, [6400*2])
    together_with_prev = np.concatenate([together_flat, prev_a])
    return together_with_prev

xform = xform_full
state_dim, action_dim = 6400*2+2, 2
actions = [2, 3]

def xform_action(a):
  if a == 2: return np.array([1.0, 0.0])
  if a == 3: return np.array([0.0, 1.0])

stateless_agent = StatelessAgent("bob", state_dim, action_dim, xform, xform_action, actions, learning_rate = 0.001, num_hidden=500)

stateless_agent.restore_model("./models/xform_full_500hidden.ckpt")

ctr = 0
times_explore = 1
while True:
  ctr += 1
  print ctr
  
  # test the agent
  # state = get_random_state(env, start_state)
  # create a batch of trace in it
  generate_pong_trace(env, start_state, stateless_agent, n=10000)
