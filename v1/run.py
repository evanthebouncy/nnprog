def run_1(state, policy, transition):
  action = policy(state)
  return transition[action](state)
 
def run_star(state, policy, transition, bnd=100):
  for i in range(bnd):
    print state
    action = policy(state)
    print action
    if action == "end":
      return state
    state = transition[action](state)
    
