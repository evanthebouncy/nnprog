import env
import learning_agent

agent_conc = learning_agent.TDLearn(env.ACTIONS, False)
agent_abst = learning_agent.TDLearn(env.ACTIONS, True)

for i in range(2000):
  s = env.gen_s()

  tr_conc = env.get_trace(agent_conc, s=s)
  agent_conc.learn(tr_conc)

  tr_abst = env.get_trace(agent_abst, s=s)
  agent_abst.learn(tr_abst)
  
agent_refi = learning_agent.avg_refine(agent_conc, env.abstract)
# 

# print "abstract Q"
# env.print_Q(agent_abst.Q)
# print "refined Q"
# env.print_Q(agent_refi.Q)

agent_conc.explore_rate = 0.0
agent_abst.explore_rate = 0.0
agent_refi.explore_rate = 0.0

ctr_conc = 0
ctr_abst = 0
ctr_refi = 0
for i in range(100):
  s = env.gen_s()
  tr_conc = env.get_trace(agent_conc, s=s)
  tr_abst = env.get_trace(agent_abst, s=s)
  tr_refi = env.get_trace(agent_refi, s=s)

  # print "conc ", tr_conc
  # print "abst ", tr_abst
  # print "refi ", tr_refi

  if tr_conc[-1][-1] == 1.0:
    ctr_conc += 1
  if tr_abst[-1][-1] == 1.0:
    ctr_abst += 1
  if tr_refi[-1][-1] == 1.0:
    ctr_refi += 1

print "concrete success ", ctr_conc, "abstract success ", ctr_abst, "refined success ", ctr_refi
