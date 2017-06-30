import env
import learning_agent

def test_bitdouble():
  bitdouble = env.BitDouble()
  policy = {'0110': 'c1', '0111': '++', '0000': '++', '0001': 'o0', '0011': '++', '0010': 'o0', '0101': 'c1', '0100': '++', '1111': '++', '1110': '++', '1100': '++', '1101': 'o1', '1010': '++', '1011': 'c1', '1001': 'c1', '1000': 'c0'}
  print len(policy.keys())
  print bitdouble
  print bitdouble.STATES
  print bitdouble.ACTIONS
  bitdouble.L = 5
  agent_angl = learning_agent.AngelicLearn(bitdouble)
  agent_angl.to_abs = policy

  success = 0
  for i in range(1):
    s = bitdouble.gen_s()
    tr_angl = bitdouble.get_trace(agent_angl, s=s)
    for ttt in tr_angl:
      print ttt
    if tr_angl[-1][-1] == 1.0:
      success += 1
    print [x[1] for x in tr_angl]
  print success
  


test_bitdouble()
