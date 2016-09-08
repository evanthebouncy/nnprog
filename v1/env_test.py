from env import *

# create env
print "test env creation"
s0 = gen_s0()
print s0

# test forward moving pt1
print "test forward moving pt1"
s0 = gen_s0()
for i in range(10):
  print s0
  s0 = pt1_plus(s0)

# test backward moving pt1
print "test backward moving pt1"
for i in range(10):
  print s0
  s0 = pt1_minu(s0)

# test forward moving pt2
print "test forward moving pt2"
s0 = gen_s0()
for i in range(10):
  print s0
  s0 = pt2_plus(s0)

# test backward moving pt2
print "test backward moving pt2"
for i in range(10):
  print s0
  s0 = pt2_minu(s0)

# test swaping
print "test swapping"
s0 = gen_s0()
s0 = pt2_plus(s0)
s0 = pt2_plus(s0)
s0 = pt2_plus(s0)
print s0
s0 = swp(s0)
print s0
s0 = swp(s0)
print s0

# test pc setting
print "test pc setting"
s0 = gen_s0()
print s0
s0 = set_pc4(s0)
print s0
s0 = set_pc3(s0)
print s0
s0 = set_pc2(s0)
print s0
s0 = set_pc1(s0)
print s0

# test abstraction
print "test abstraction"
s0 = gen_s0()
for i in range(7):
  print s0
  print abstract(s0)
  s0 = pt2_plus(s0)

s0 = swp(s0)
print s0
print abstract(s0)
