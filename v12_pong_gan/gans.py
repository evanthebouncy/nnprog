import gym
import gym.envs.atari
import random
import sys
from utils import *

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
    if a == 2: return [1.0, 0.0]
    if a == 3: return [0.0, 1.0]

  state_dim, action_dim = 6400, 2
  actions = [2, 3]

  stateless_agent = StatelessAgent("bob", state_dim, action_dim, xform, xform_action, actions)
  agent_sas = sample_agent_sa(stateless_agent)

  from discriminator import *

  # planner_sas = planner_sas[:1]
  # agent_sas = agent_sas[:1]

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

  
