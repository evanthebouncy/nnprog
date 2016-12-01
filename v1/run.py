def run_1(state, policy, transition):
  action = policy(state)
  return transition[action](state)
 
def run_star(state, policy, transition, bnd=20):
  for i in range(bnd):
    print "state:"
    print state
    action = policy(state)
    print "chosen action:"
    print action
    if action == "end":
      return state
    state = transition[action](state)
    
