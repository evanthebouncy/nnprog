from env import *
import pickle

L = 32
bugzero_test = BugZero(L)
mazes = []
for i in range(1000):
  maze = bugzero_test.gen_s()
  mazes.append(maze)

pickle.dump( mazes, open( "test_datas/mazes.p", "wb" ) )

