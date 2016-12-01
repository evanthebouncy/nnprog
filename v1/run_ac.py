from ac_agent import *

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
    r = -0.0001
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
    
actor = Actor(Actions)    
critic = Critic(Actions)

bndd = 4

lots_trace = []
# trianing
for bnd in range(bndd, bndd+1):
  for i in range(100 * bnd * bnd * bnd):
    print i
    s = gen_s0(bnd,bnd,4,det=True)
    trace = run_star(s, actor, ActionsMap)
    lots_trace += trace
    critic.learn(trace)
    if i % 20 == 0:
      actor.learn(lots_trace, critic)
      lots_trace = []
      

# testing...
actor.det = True
s = gen_s0(bnd,bnd,4,det=True)
s_start = s
trace = run_star(s, actor, ActionsMap)
print "Trace"
for tr in trace:
  s,a,ss,r = tr
  print "\n\n======================"
  print "state ", s
  print "abs state ", abstract(s)
  print "action ", a
  print "\nactor_q a", actor.Q[abstract(s), a]
  print "all actions: "
  for b in actor.actions:
    print b, actor.Q[abstract(s),b]
  print "\ncritic_q a", critic.Q[stringify(s),a]
  print "all actions: "
  for b in critic.actions:
    print b, critic.Q[stringify(s),b]

print "visited these many distinct states"
print len(actor.Q)
