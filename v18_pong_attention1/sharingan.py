from stateless_model import *
import random

class Sharingan:
  
  def __init__(self):
    print "NANI?!"
    # self.sharingan = StatelessAgent("look", 100, 100, 

  def random_look(self, sub_img):
    return random.randint(0, 99)

  def rand_move(self, sub_img, glimpse):
    return random.choice([2,3])

  def act(self, sssa, show_prob = False):
    s_ss, a = sssa
    s, ss = s_ss
    
    
    
