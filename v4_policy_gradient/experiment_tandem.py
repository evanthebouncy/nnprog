from env import *
from draw import *
from td_agent import *
from stateless_model import *

def run_knight():
  knight = Knight()
  td_agent = TDLearn(knight)
  driver = StatelessAgent("driver", 3, 3)
  tandem = TandemAgent(td_agent, driver, "abst")
  exp = Experience(100)


  # populate some experience
  for i in xrange(100):
    tr = knight.get_trace(tandem)
    exp.add(tr)

  # do some training
  for i in xrange(1000000):
    tr = knight.get_trace(tandem)
    # print tr
    exp.add(tr)

    # training
    trace_batch = exp.n_sample(20)
    tandem.learn(trace_batch, knight)

    if i == 200:
      tandem.set_abst(["L"])
      for i in xrange(100):
        tr = knight.get_trace(tandem)
        exp.add(tr)
    if i == 400:
      tandem.set_abst(["L", "R"])
      for i in xrange(100):
        tr = knight.get_trace(tandem)
        exp.add(tr)
    if i == 600:
      tandem.set_abst(["L", "R", "U"])
      for i in xrange(100):
        tr = knight.get_trace(tandem)
        exp.add(tr)

    if i % 100 == 0:
      print i, tr[-1]
      print tr

def run_knight():
  knight = Knight()
  td_agent = TDLearn(knight)
  driver = StatelessAgent("driver", 3, 3)
  tandem = TandemAgent(td_agent, driver, "abst")
  exp = Experience(100)


  # populate some experience
  for i in xrange(100):
    tr = knight.get_trace(tandem)
    exp.add(tr)

  # do some training
  for i in xrange(1000000):
    tr = knight.get_trace(tandem)
    # print tr
    exp.add(tr)

    # training
    trace_batch = exp.n_sample(20)
    tandem.learn(trace_batch, knight)

    if i == 200:
      tandem.set_abst(["L"])
      for i in xrange(100):
        tr = knight.get_trace(tandem)
        exp.add(tr)
    if i == 400:
      tandem.set_abst(["L", "R"])
      for i in xrange(100):
        tr = knight.get_trace(tandem)
        exp.add(tr)
    if i == 600:
      tandem.set_abst(["L", "R", "U"])
      for i in xrange(100):
        tr = knight.get_trace(tandem)
        exp.add(tr)

    if i % 100 == 0:
      print i, tr[-1]
      print tr
