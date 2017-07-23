from env import *
from draw import *
from td_agent import *
import pickle
import math

EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 10000

def get_accuracy(agent, env):
  acc = 0.0
  for jj in range(100):
    tr = env.get_trace(agent)
    if tr[-1][-1] == 1.0: acc += 1.0
  return acc / 100

def collect_stat(record, trace, env):
  for tr in trace:
    s, name_a, _, _ = tr
    _, a = name_a
    if (env.abstract(s), a) not in record:
      record[(env.abstract(s), a)] = 0
    record[(env.abstract(s), a)] += 1

bitadd = BitAdd()
td_agent = TDLearn(bitadd)
exp = Experience(2000)

# set explroation rates
td_agent.explore_rate = EPS_START 

# populate some experience
for i in xrange(2000):
  tr = bitadd.get_trace(td_agent)
  # if tr[-1][-1] == 1.0: print tr[-1]
  exp.add(tr)

record = {}

# do some training
for i in xrange(10000000):

  tr = bitadd.get_trace(td_agent)
  # print tr
  exp.add(tr)

  # training
  trace_batch = exp.n_sample(20)
  td_agent.learn(trace_batch, bitadd)

  if tr[-1][-1] == 1.0: print tr[-1]

  if i % 1000 == 0:
    # set exploration to 0 to get accuracy
    td_agent.explore_rate = 0.0
    acc = get_accuracy(td_agent, bitadd)
    # adjust episilon
    epi = EPS_START * (1 - acc) + EPS_END * acc
    print epi
    td_agent.explore_rate = epi

    print "table state accuracy {0} at {1} with epi {2} ".format(acc, i,epi)
    print i, tr, len(td_agent.Q)
    pickle.dump(td_agent, open("bitadd_q.p", "wb"))

# knight = Knight()
# td_agent = TDLearn(knight)
# exp = Experience(1000)
# 
# # populate some experience
# for i in xrange(100):
#   tr = knight.get_trace(td_agent)
#   exp.add(tr)
# 
# record = {}
# 
# # do some training
# for i in xrange(10000):
#   tr = knight.get_trace(td_agent)
#   # print tr
#   exp.add(tr)
# 
#   # training
#   trace_batch = exp.n_sample(20)
#   td_agent.learn(trace_batch, knight)
# 
#   print i, tr[-1]
# 
#   if i > 500:
#     collect_stat(record, tr, knight)
# 
#   if i % 100 == 0:
#     print "record"
#     for blah in sorted(list(record.keys())):
#       print blah, record[blah]

