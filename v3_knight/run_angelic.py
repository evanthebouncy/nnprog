import env
import learning_agent

def run_knight():
  knight = env.Knight()
  print knight
  print knight.STATES
  print knight.ACTIONS
  policy = learning_agent.synthesize(knight, learning_agent.AngelicLearn)
  print policy

# run_knight()

def run_flag():
  flag = env.Flag()
  print flag
  print flag.STATES
  print flag.ACTIONS
  policy = learning_agent.synthesize(flag, learning_agent.AngelicLearn)
  print policy

# run_flag()

def run_bitdouble():
  bitdouble = env.BitDouble()
  print bitdouble
  print bitdouble.STATES
  print bitdouble.ACTIONS
  policy = learning_agent.synthesize(bitdouble, learning_agent.AngelicLearn, 
                                     train_iter=5000, test_iter=500)
  print policy

run_bitdouble()
