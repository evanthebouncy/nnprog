from gans import *
from stateless_model import *
from discriminator import *
from draw import *

state_dim, action_dim = 6400, 2
actions = [2, 3]

def xform(state):
  ob1, ob2 = state
  if ob1 is None or ob2 is None:
    return np.zeros([6400])
  else:
    ob1, ob2 = get_signal_full_image(ob1, ob2)
    diff = ob2 - ob1
    diff = diff[:, 40:80]
    half_ob2 = ob2[:, 40:80]
    #draw(diff, "diff.png")
    #draw(half_ob2, "half_ob2.png")
    together = np.concatenate([diff, half_ob2], axis=1)
    #draw(together, "together.png")
    #assert 0
    
    return np.reshape(together, [6400])

def xform_action(a):
  if a == 2: return np.array([1.0, 0.0])
  if a == 3: return np.array([0.0, 1.0])

stateless_agent = StatelessAgent("bob", state_dim, action_dim, xform, xform_action, actions)
discrim = Discriminator("dis", state_dim, action_dim, xform, xform_action)

ctr = 0
while True:
  ctr += 1
  print ctr
  
  do_render = True if ctr % 10 == 0 else False

  # train the discriminator
  discrim.train_discrim(sample_planner_sa(), sample_agent_sa(stateless_agent))
  # train the agent
  state = get_random_state(env, start_state)
  # create a batch of trace in it
  agent_trace = generate_pong_trace(env, state, stateless_agent, do_render=do_render)
  # augment the reward signal with discriminator reward
  agent_trace_dagger = dagger_reward(state, agent_trace)
  # usual policy gradient on the augmented trace
  stateless_agent.learn_policy_grad([agent_trace_dagger])

  
