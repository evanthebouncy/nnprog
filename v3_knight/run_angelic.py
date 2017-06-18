import env
import learning_agent

policy = learning_agent.synthesize(["U", "L", "R"], env.ACTIONS, learning_agent.AngelicLearn, env)
print policy

# agent_angl = learning_agent.AngelicLearn(env.ACTIONS)
# agent_angl.to_abs["L"] = "UP"
# agent_conc = learning_agent.TDLearn(env.ACTIONS, False)
# 
# for i in range(2000):
#   s = env.gen_s()
# 
#   tr_angl = env.get_trace(agent_angl, s=s)
#   agent_angl.learn(tr_angl)
# 
#   tr_conc = env.get_trace(agent_conc, s=s)
#   agent_conc.learn(tr_conc)
#   
# 
# agent_angl.explore_rate = 0.0
# agent_conc.explore_rate = 0.0
# 
# ctr_angl = 0
# ctr_conc = 0
# 
# for i in range(100):
#   s = env.gen_s()
#   tr_angl = env.get_trace(agent_angl, s=s)
#   tr_conc = env.get_trace(agent_conc, s=s)
# 
#   # print "angl ", tr_angl
#   # print "actor ", tr_actor
#   # print "refi ", tr_refi
# 
#   if tr_angl[-1][-1] == 1.0:
#     ctr_angl += 1
#   if tr_conc[-1][-1] == 1.0:
#     ctr_conc += 1
# 
# print "anglic ", ctr_angl, " concrete ", ctr_conc
# print " Q"
# env.print_Q(agent_angl.Q)
