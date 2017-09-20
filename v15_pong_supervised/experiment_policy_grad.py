from supervise import *
from stateless_model import *
from discriminator import *
from draw import *

def xform_extreme(state):
  ob1, ob2, prev_a = state
  prev_a = xform_action(prev_a)
  simple_sig = get_up_down_signal(ob1, ob2)
  return simple_sig

def xform_compact(state):
  def get_around_center(img, xy):
    x, y = xy
    # hack some clips
    x = max(min(x, 78), 1)
    y = max(min(y, 78), 1)
    return np.reshape( img[x-1:x+2, y-1:y+2], [9])

  ob1, ob2, prev_a = state
  prev_a = xform_action(prev_a)
#  return get_simple_signal(ob1, ob2)
  if ob1 is None or ob2 is None:
    return np.zeros([state_dim])

  else:
    ball = get_ball(ob2) if get_ball(ob2) is not None else (5,5)
    paddle = get_our_paddle(ob2) if get_our_paddle(ob2) is not None else (5,5)

    ob1, ob2 = get_signal_full_image(ob1, ob2)
    diff = ob2 - ob1

    ob2_ball = get_around_center(ob2, ball)
    ob2_paddle = get_around_center(ob2, paddle)
    diff_ball = get_around_center(diff, ball)
    diff_paddle = get_around_center(diff, paddle)

    normalize_ball = np.array([ball[0], ball[1]], np.float32) / 80.0
    normalize_paddle = np.array([paddle[0], paddle[1]], np.float32) / 80.0

    # print normalize_ball, normalize_paddle, "ball, padle"

    vect_diff = normalize_ball - normalize_paddle

    #draw(diff, "diff.png")
    #draw(half_ob2, "half_ob2.png")
    together_flat = np.concatenate([ob2_ball, ob2_paddle, diff_ball, diff_paddle, vect_diff, normalize_paddle])
    #draw(together, "together.png")
    #assert 0
    
    together_with_prev = np.concatenate([together_flat, prev_a])
    # together_with_prev = together_flat
    return together_with_prev

def xform_semi(state):
  ob1, ob2, prev_a = state
  prev_a = xform_action(prev_a)
#  return get_simple_signal(ob1, ob2)
  if ob1 is None or ob2 is None:
    return np.zeros([882])
  else:
    ob1, ob2 = get_signal_full_image(ob1, ob2)
    diff = ob2 - ob1
    diff = diff[:, 60:70]
    half_ob2 = ob2[:, 70:71]
    #draw(diff, "diff.png")
    #draw(half_ob2, "half_ob2.png")
    together = np.concatenate([diff, half_ob2], axis=1)
    #draw(together, "together.png")
    #assert 0
    
    together_flat = np.reshape(together, [880])
    together_with_prev = np.concatenate([together_flat, prev_a])
    return together_with_prev

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

xform = xform_extreme
state_dim, action_dim = 3, 2
actions = [2, 3]


def xform_action(a):
  if a == 2: return np.array([1.0, 0.0])
  if a == 3: return np.array([0.0, 1.0])

stateless_agent = StatelessAgent("bob", state_dim, action_dim, xform, xform_action, actions, learning_rate = 0.01, num_hidden=10)

ctr = 0
times_explore = 1
while True:
  ctr += 1
  print ctr
  
  do_render = True if ctr % 10 == 0 else False

#   if ctr % 10 == 0:
#     planner_sa = [sample_planner_sa()]
#     stateless_agent.learn_supervised(planner_sa)
  

  # test the agent
  state = get_random_state(env, start_state)
  # create a batch of trace in it
  agent_trace_batch = [generate_pong_trace(env, state, stateless_agent, do_render=do_render)\
                       for _ in range(times_explore)]

  for agr in agent_trace_batch:
    rewards = 0.0
    for s,a,r in agr:
      rewards += r
    print "explore trace reward ", rewards

  stateless_agent.learn_policy_grad(agent_trace_batch)

  

  
