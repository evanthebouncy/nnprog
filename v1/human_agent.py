from env import *

def human_policy(s):
  abs_s = abstract(s)
  
  the_action = set()

  print "cur_state", abs_s
  # in the first phase increment the pt2
  if abs_s[0][3] != 0 and abs_s[1][0] == 1:
    the_action.add("pt2_plus")

  # end first phase and go to second phase
  if abs_s[0][3] == 0 and abs_s[1][0] == 1:
    the_action.add("set_pc2")

  if abs_s[0][4] == 0 and (abs_s[2][4] == 1 or abs_s[2][7] == 1):
    the_action.add("swp")

  if abs_s[2][5] == 1 and abs_s[0][4] == 0:
    the_action.add("pt1_plus")

  if abs_s[2][1] == 1 and abs_s[0][4] == 0:
    the_action.add("pt2_minu")


  if abs_s[1][1] == 1 and abs_s[0][4] == 1:
    the_action.add("end")

  print the_action
  assert len(the_action) == 1
  return list(the_action)[0]

