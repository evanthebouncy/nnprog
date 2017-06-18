import env
import learning_agent

agent_crit = learning_agent.TDLearn(env.ACTIONS, False)
agent_actor = learning_agent.TDLearn(env.ACTIONS, True)
agent_conc = learning_agent.TDLearn(env.ACTIONS, False)

for i in range(2000):
  s = env.gen_s()

  tr_actor = env.get_trace(agent_actor, s=s)
  agent_crit.learn(tr_actor)
  agent_actor.learn_with_critic(tr_actor, agent_crit)

  tr_conc = env.get_trace(agent_conc, s=s)
  agent_conc.learn(tr_conc)
  
# print "actorract Q"
# env.print_Q(agent_actor.Q)

agent_crit.explore_rate = 0.0
agent_actor.explore_rate = 0.0
agent_conc.explore_rate = 0.0

ctr_crit = 0
ctr_actor = 0
ctr_conc = 0

for i in range(100):
  s = env.gen_s()
  tr_crit = env.get_trace(agent_crit, s=s)
  tr_actor = env.get_trace(agent_actor, s=s)
  tr_conc = env.get_trace(agent_conc, s=s)

  # print "crit ", tr_crit
  # print "actor ", tr_actor
  # print "refi ", tr_refi

  if tr_crit[-1][-1] == 1.0:
    ctr_crit += 1
  if tr_actor[-1][-1] == 1.0:
    ctr_actor += 1
  if tr_conc[-1][-1] == 1.0:
    ctr_conc += 1

print "actor ", ctr_actor, " critic ", ctr_crit, " concrete ", ctr_conc
