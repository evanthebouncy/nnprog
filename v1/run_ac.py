from ac_agent import *

# testing...
def testing_debug():
  actor.det = True
  s = gen_s0(bnd_r,bnd_r,2,det=True)
  s_start = s
  trace = run_star(s, actor, ActionsMap)
  print "\n\n\n ##################################################### Trace"
  for tr in trace:
  # for tr in trace:
    s,a,ss,r = tr
    print "\n\n======================"
    print "state ", s
    print "abs state ", abstract(s)
    print "action ", a
    print "reward ", r

    print "\nactor_q a", actor.Q[abstract(s), a]
    print "responsible for learning this... "
    if (abstract(s), a) in actor.responsible:
      print actor.responsible[(abstract(s), a)]
    print "all actions: "
    for b in actor.actions:
      print b, actor.Q[abstract(s),b]
    print "\ncritic_q a", critic.Q[stringify(s),a]
    print "all actions: "
    for b in critic.actions:
      print b, critic.Q[stringify(s),b]

  print "visited these many distinct states (actor)"
  print len(actor.Q)
  print "visited these many distinct states (critic)"
  print len(critic.Q)


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

bnd = 6

lots_trace = []
# trianing
for i in range(1000000):
  actor.det = False
#  print i

#  bnd_r = random.randint(2,bnd)
  bnd_r = bnd
  if bnd_r < 4:
    bnd_r = random.randint(2,bnd)
  s = gen_s0(bnd_r,bnd_r,2,det=True)
  trace = run_star(s, actor, ActionsMap)
  lots_trace += trace

  critic.learn(trace)
  if i % 100 == 19:
    actor.learn(lots_trace, critic)
    lots_trace = []

  if i % 1000 == -1 % 1000:
    try:
      testing_debug()
    except:
      pass
      

# print "unexplored actors"
# for s,a in actor.Q:
#   if actor.Q[(s,a)] == 0.0:
#     print s, a, actor.Q[(s,a)]
# 
# print "unexplored critic"
# for s,a in critic.Q:
#   sss = eval(s)
#   if a == "end" and sss[0] == [2,1]:
#     print s, a, critic.Q[(s,a)]
# 
