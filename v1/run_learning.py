from learning_agent import *

# takes an agent, runs it agains the interpreter, give a trace
def run_star(s_start, agent, transition, show_step=False, bnd=20):
  s_final = list(reversed(s_start["ary"]))

  tr = []
  s = s_start
  for i in range(bnd):
    action = agent.get_move(s)
    if show_step:
      print "state:"
      print s, abstract(s)
      print "action:"
      print action
    ss = transition[action](s)
    # give a slight negative reward for doing nothing and to force exploration
    r = -0.00001
    # reward for reaching final state
    if s["ary"] == s_final and action == "end":
      r = 1.0
    if s["ary"] != s_final and action == "end":
      r = -1.0
    tr.append((s,action,ss,r))
    if action == "end":
      return tr
    # update state...
    s = ss
  return tr
    
    
learner = TDLearn(Actions, True)
# learner = TDLearn(Actions, False)
learner.explore_rate = 0.1

bnd = 4

for bnd in range(bnd, bnd+1):
  for i in range(10000 * bnd):
    print i
    s = gen_s0(bnd,bnd,4,det=True)
    trace = run_star(s, learner, ActionsMap)
    learner.learn(trace)

# testing...
learner.explore_rate = 0.0
s = gen_s0(bnd,bnd,4,det=True)
s_start = s
print s, abstract(s)
trace = run_star(s, learner, ActionsMap, show_step = True)
print "Trace"
for tr in trace:
  s,a,ss,r = tr
  print "======================"
  print s
  print a

print "visited these many distinct states"
print len(learner.Q)

print "Q so far on first move"
print "first state is: ", abstract(s_start)
for a in learner.actions:
  print a, learner.Q[stringify(s_start),a]

if trace[-1][-1] > 0:
  print "success!"
  print "abstr summary :"
  for tr in trace:
    s,a,ss,r = tr
    print abstract(s), a

# # s = gen_s0(5,5,4)
# # trace = run_star(s, learner, ActionsMap) 
# # print map(lambda x: (x[0],x[1],x[3]), trace)
# # print learner.Q
