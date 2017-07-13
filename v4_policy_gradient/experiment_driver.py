from env import *
from draw import *
from knight_model import *

L = 6

knight = Knight()
driver = Driver()
exp = Experience(1000)

# populate some experience
for i in xrange(100):
  tr = knight.get_trace(driver)
  exp.add(tr)

# do some training
for i in xrange(10000):
  tr = knight.get_trace(driver)
  # print tr
  exp.add(tr)

  # training
  trace_batch = exp.n_sample(20)
  driver.learn(trace_batch, knight)

