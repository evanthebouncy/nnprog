from run_learning import *

def collect_good_tr(agentMkr, transition):
  good_tr = []  
  for bnd in range(2,5):
    for i in range(1000):
      learner = agentMkr(Actions, True)
      for i in range(100 * bnd * bnd * bnd):
        print i
        s = gen_s0(bnd,bnd,4,det=True)
        trace = run_star(s, learner, transition)
        learner.learn(trace)

      # testing...
      learner.explore_rate = 0.0
      s = gen_s0(bnd,bnd,4,det=True)
      s_start = s
      trace = run_star(s, learner, transition)

      if trace[-1][-1] > 0:
        good_tr.append(trace)
  return good_tr
      
      
good_tr = collect_good_tr(TDLearn, ActionsMap)
print len(good_tr)

good_tr_str = repr(good_tr)
fd = open("tr.good","w")
fd.write(good_tr_str)
fd.close()

