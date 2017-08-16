from env import *
from draw import *
from td_agent import *
from oracle_model import *
from stateless_model import *
import pickle
import sys
from human_agent import *
import pickle

test_mazes = pickle.load(open("test_datas/mazes.p","rb"))
bugzero = BugZero(16)

manbug = Human1(bugzero)

print get_accuracy_from_set(manbug, bugzero, test_mazes)
