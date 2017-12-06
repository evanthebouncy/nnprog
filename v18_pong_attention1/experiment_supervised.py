from supervise import *
from stateless_model import *
from subsample import *
from draw import *

state_dim, action_dim = 402, 2
actions = [2, 3]

def padpad(ob):
  aline = np.reshape( ob[:, 70:71], [80])
  our_paddle_y = np.argmax(aline)
  paddlex, paddley = 70, min(max(10, our_paddle_y + 4), 70)
  # print paddlex, paddley
  pad_glimpse = ob[paddley-10:paddley+10, paddlex-8:paddlex+2]
  return pad_glimpse


def xform_subsample(state):
  ob1_ob2, prev_a = state
  ob1, ob2 = ob1_ob2

  prev_a = xform_action(prev_a)
  if ob1 is None or ob2 is None:
    return np.zeros([402])
  else:

    pad1, pad2 = padpad(ob1), padpad(ob2)

    # print pad1
    # print " " 
    # print pad2
    # assert 0

    # pad1 = ob1[:, 65:70]
    # pad2 = ob2[:, 65:70]

    ob1, ob2 = dim_reduce(ob1, 3), dim_reduce(ob2, 3)
    together = np.concatenate([ob1, ob2], axis=1)

    pad_diff = np.reshape(pad1 - pad2, [200])
    
    # pad1_flat = np.reshape(pad1, [100])
    # pad2_flat = np.reshape(pad2, [100])

    together_flat = np.reshape(together, [100*2])
    # together_with_prev = np.concatenate([together_flat, pad1_flat, pad2_flat, prev_a])
    together_with_prev = np.concatenate([together_flat, pad_diff, prev_a])
    return together_with_prev

xform = xform_subsample

def xform_action(a):
  if a == 2: return np.array([1.0, 0.0])
  if a == 3: return np.array([0.0, 1.0])

stateless_agent = StatelessAgent("bob", state_dim, action_dim, xform, xform_action, actions)

ctr = 0
times_explore = 1
while True:
  ctr += 1
  print ctr
  
  do_render = True if ctr % 10 == 0 else False

  if ctr % 100 == 0:
    stateless_agent.save_model("models/pongpong.ckpt")

  # train the discriminator
  planner_sa = [sample_planner_sa()]
  stateless_agent.learn_supervised(planner_sa)

  # test the agent
  if do_render:
    state = get_random_state(env, start_state)
    # create a batch of trace in it
    agent_trace_batch = [generate_pong_trace(env, state, stateless_agent, do_render=do_render)\
                         for _ in range(times_explore)]

  # for agr in agent_trace_batch:
  #   rewards = 0.0
  #   for s,a,r in agr:
  #     rewards += r
  #   print "explore trace reward ", rewards

  # stateless_agent.learn_policy_grad(agent_trace_batch)

