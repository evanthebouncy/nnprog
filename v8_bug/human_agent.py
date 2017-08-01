import random

def go_to(xform, s):
  glimpse, goal_dir, path = xform(s)
  return find_closest_move(goal_dir)

def find_closest_move(intended_dir):
  up =    [0.0, -1.0], "U"
  left =  [-1.0, 0.0], "L" 
  down =  [0.0, 1.0],  "D" 
  right = [1.0, 0.0], "R" 

  def dot(aa,bb):
    return aa[0] * bb[0] + aa[1] * bb[1]

  return max([(dot(x[0], intended_dir), x[1]) for x in [up,left,down,right]])[1] 

def find_closest_moves(xform, s, intended_dir):
  up =    [0.0, -1.0], "U"
  left =  [-1.0, 0.0], "L" 
  down =  [0.0, 1.0],  "D" 
  right = [1.0, 0.0], "R" 

  def dot(aa,bb):
    return aa[0] * bb[0] + aa[1] * bb[1]

  rankz = [(dot(x[0], intended_dir), x[1]) for x in [up,left,down,right]]
  rankz.sort()
  rankz = [r[1] for r in rankz]
  return rankz

# perform a cww twist
def twist(dirr):
  if dirr == "U": return "L"
  if dirr == "L": return "D"
  if dirr == "D": return "R"
  if dirr == "R": return "U"

# follow obsticle in a clockwise manner
def follow_obsticle(xform, s):
  glimpse, goal_dir, path = xform(s)
  # find the average of the obsticle's direction
  odirx, odiry = 0.0, 0.0
  for xx in range(3):
    for yy in range(3):
      if glimpse[yy][xx] == 1.0:
        odirx += xx - 1
        odiry += yy - 1
  # print glimpse
  ob_dirs = find_closest_moves(xform, s, (odirx, odiry))
  # print "obsticle directions ", ob_dirs, " from ", odirx, odiry
  twists = [twist(ob_dir) for ob_dir in ob_dirs]
  twists = [t for t in twists if can_advance_in_dir(xform, s, t)]
  twists = twists[:2]
  return random.choice(twists)
  

def can_advance_in_dir(xform, s, dirr):
  glimpse, goal_dir, path = xform(s)
  if dirr == "U": return glimpse[0][1] != 1.0
  if dirr == "L": return glimpse[1][0] != 1.0
  if dirr == "R": return glimpse[1][2] != 1.0
  if dirr == "D": return glimpse[2][1] != 1.0
  assert 0

def get_random_move(xform, s):
  dirr = random.choice(["U", "L", "R", "D"])
  if can_advance_in_dir(xform, s, dirr):
    return dirr
  else:
    return get_random_move(xform, s)

def get_next_pos(loc, dirr):
  cpx, cpy = loc
  if dirr == "U": return cpx, cpy-1
  if dirr == "L": return cpx-1, cpy
  if dirr == "R": return cpx+1, cpy
  if dirr == "D": return cpx, cpy+1
  assert 0
  
def can_advance(xform, s):
  glimpse, goal_dir, path = xform(s)
  dirr = go_to(xform, s)
  can_go = can_advance_in_dir(xform, s, dirr)
  return can_go 

def cycle_detection(position, path):
  if len(path) == 0:
    return False
  return position in [p[0] for p in path]

def current_move_in_cycle(xform, s, dirr):
  glimpse, goal_dir, path = xform(s)
  start = s[1]
  nxt_loc = get_next_pos(start, dirr)
  if len(path) == 0: return False
  else:
    locations = [p[0] for p in path]
    return nxt_loc in locations[:-1]
  

def flail(xform, s):
  glimpse, goal_dir, path = xform(s)
  moves = ["U","L","R","D"]
  flail_moves = [t for t in moves if can_advance_in_dir(xform, s, t)]
  return random.choice(flail_moves)
    
class Human1:

  def __init__(self, bugzero):
    self.bugzero=bugzero

  def xform(self, s):
    bugzero = self.bugzero
    maze, start, end, path, call_state = s
    center_state = bugzero.centered_state(s)
    glimpse = bugzero.get_glimpse(center_state, 1)
    scent = bugzero.get_scent(center_state, 1)
    goal_dir = bugzero.get_goal_direction(center_state)
    return glimpse, goal_dir, path

  def act(self, s):
#    print s[1]
    if can_advance(self.xform, s):
      dirr = go_to(self.xform, s)
#      print "goto ", dirr
    else:
      dirr = follow_obsticle(self.xform, s)
#      print "follow ", dirr

    if current_move_in_cycle(self.xform, s, dirr):
#      print "flail "
      dirr = flail(self.xform, s)
#      print dirr

    return "human", dirr, 0.0
