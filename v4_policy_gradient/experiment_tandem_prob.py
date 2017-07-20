from env import *
from draw import *
from td_agent import *
from stateless_model import *

double = BitDouble()
td_agent = TDLearn(double)
driver = StatelessAgent("driver", len(double.STATES), len(double.ACTIONS))
tandem = TandemAgent(td_agent, driver, "prob")
exp = Experience(100)

tandem.set_prob(0.00)
# populate some experience
for i in xrange(100):
  tr = double.get_trace(tandem)
  exp.add(tr)

# do some training
for i in xrange(1000000):
  tr = double.get_trace(tandem)
  # print tr
  exp.add(tr)

  # training
  trace_batch = exp.n_sample(20)
  tandem.learn(trace_batch, double)

  if i % 100 == 0:
    tandem.set_prob(tandem.prob - 0.01)
    print i, tandem.prob, tr[-1]
    print tr

# flag = Flag()
# td_agent = TDLearn(flag)
# driver = StatelessAgent("driver", 8, 4)
# tandem = TandemAgent(td_agent, driver, "prob")
# exp = Experience(100)
# 
# tandem.set_prob(0.00)
# # populate some experience
# for i in xrange(100):
#   tr = flag.get_trace(tandem)
#   exp.add(tr)
# 
# # do some training
# for i in xrange(1000000):
#   tr = flag.get_trace(tandem)
#   # print tr
#   exp.add(tr)
# 
#   # training
#   trace_batch = exp.n_sample(20)
#   tandem.learn(trace_batch, flag)
# 
#   if i % 100 == 0:
#     tandem.set_prob(tandem.prob - 0.01)
#     print i, tandem.prob, tr[-1]
#     print tr

# knight = Knight()
# td_agent = TDLearn(knight)
# driver = StatelessAgent("driver", 3, 3)
# tandem = TandemAgent(td_agent, driver, "prob")
# exp = Experience(100)
# 
# tandem.set_prob(0.99)
# # populate some experience
# for i in xrange(100):
#   tr = knight.get_trace(tandem)
#   exp.add(tr)
# 
# # do some training
# for i in xrange(1000000):
#   tr = knight.get_trace(tandem)
#   # print tr
#   exp.add(tr)
# 
#   # training
#   trace_batch = exp.n_sample(20)
#   tandem.learn(trace_batch, knight)
# 
#   if i % 100 == 0:
#     tandem.set_prob(tandem.prob - 0.01)
#     print i, tandem.prob, tr[-1]
#     print tr
