import gym
import gym.envs.atari
import random
import sys
from utils import *
from generate_planner_traces import *

env = gym.envs.atari.atari_env.AtariEnv(obs_type='image', frameskip=2)
start_state = env.clone_full_state()

planner_traces = pickle.load(open("datas/planner_traces.p","rb"))

class RecordAgent:
  def __init__(self, recorded_actions):
    self.actions = recorded_actions
    self.counter = 0

  # takes some observation but doesn't really use them
  def act(self, s):
    obs_prev, obs = s
    assert self.counter < len(self.actions), "ran out of actions!!"
    ret = self.actions[self.counter]
    self.counter += 1
    return ret
    
def sample_planner_sa(n_pairs = 40):
  state, r_best, best_as = random.choice(planner_traces)
  rec_agent = RecordAgent(best_as)
  trace = generate_pong_trace(env, state, rec_agent, do_render=False)
  trs = [random.choice(trace) for i in range(n_pairs)]
  return [(pz[0], pz[1]) for pz in trs]

def sample_agent_sa(agent, n_pairs = 40):
  state = get_random_state(env, start_state)
  trace = generate_pong_trace(env, state, agent, do_render=False)
  trs = [random.choice(trace) for i in range(n_pairs)]
  return [(pz[0], pz[1]) for pz in trs]

# augment the trace of the agent with the extra reward signal from the discriminator
def augment_reward(discrim, agent_trace):
  ret_trace = []
  for s,a,r in agent_trace:
    discrim_score = discrim.score(s,a) / 100.0
    # ret_trace.append((s,a,r+discrim_score)) 
    ret_trace.append((s,a,r+discrim_score)) 
  return ret_trace

def recover_full_trace(agent_start_state, agent_trace):
  env.reset()
  env.restore_full_state(agent_start_state)
  actionzzz = [2]
  recovered_state = [agent_start_state]
  for s,a,r in agent_trace:
    obs, reward, done, c = env.step(a) 
    recovered_state.append(env.clone_full_state())
    actionzzz.append(a)

  ret = []
  for i, s_a_r in enumerate(agent_trace):
    s,a,r = s_a_r
    ret.append((recovered_state[i], s[1], s[0], actionzzz[i]))
  return ret

def dagger_reward(agent_start_state, agent_trace, xform=None):
  total_tiny = 0.0
  recovered = recover_full_trace(agent_start_state, agent_trace)
  ret_trace = []
  for i in range(len(agent_trace)):
    s,a,r = agent_trace[i]
    # rec_s, s1, s0, prev_a = recovered[i]
    oracle_a = act(*recovered[i])
    tiny_reward = 0.1 if oracle_a == a else -0.1

    if xform != None:
      print "s,a,oracle_a", xform(s), a, oracle_a, tiny_reward

    ret_trace.append((s,a,tiny_reward)) 
    total_tiny += tiny_reward
  print "total tiny reward ", total_tiny
  return ret_trace
    
    

if __name__ == "__main__":
  planner_sas = sample_planner_sa()

  from stateless_model import *
  # generate a fake agent trace . . .
  def xform(state):
    ob1, ob2 = state
    if ob1 is None or ob2 is None:
      return np.zeros([6400])
    else:
      ob1, ob2 = get_signal_full_image(ob1, ob2)
      return np.reshape(ob2 - ob1, [6400])

  def xform_action(a):
    if a == 2: return np.array([1.0, 0.0])
    if a == 3: return np.array([0.0, 1.0])

  state_dim, action_dim = 6400, 2
  actions = [2, 3]

  stateless_agent = StatelessAgent("bob", state_dim, action_dim, xform, xform_action, actions)
  agent_sas = sample_agent_sa(stateless_agent)

  from discriminator import *

  for i in range(40):
    print planner_sas[i][1] == agent_sas[i][1]

  discrim = Discriminator("dis", state_dim, action_dim, xform, xform_action)

  prev_planner_score = sum([discrim.score(*sa_p) for sa_p in planner_sas])
  prev_agent_score = sum([discrim.score(*sa_p) for sa_p in agent_sas])
  discrim.train_discrim(planner_sas, agent_sas)
  post_planner_score = sum([discrim.score(*sa_p) for sa_p in planner_sas])
  post_agent_score = sum([discrim.score(*sa_p) for sa_p in agent_sas])

  print prev_planner_score, post_planner_score, " this should go UP "
  print prev_agent_score, post_agent_score, " this should go DOWN "

  state = get_random_state(env, start_state)
  # create a batch of 1 thing in it
  agent_trace_batch = [generate_pong_trace(env, state, stateless_agent, do_render=False)]
  agent_trace_batch_aug = [augment_reward(discrim, agent_trace)\
                           for agent_trace in agent_trace_batch]

  for i in range(100):
    print agent_trace_batch[0][i][2], agent_trace_batch_aug[0][i][2]

  stateless_agent.learn_policy_grad(agent_trace_batch_aug)

  
