import numpy as np
from numpy import array
from scipy.misc import imresize
import copy
from scipy.ndimage.filters import gaussian_filter


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_mnist_img(L, test=False):
  for i in range(np.random.randint(1, 1000)):
    img, _x = mnist.train.next_batch(1)
    if test:
      img, _x = mnist.test.next_batch(1)

  img = np.reshape(img[0], [28,28])
  # rescale the image to 14 x 14
  # img = imresize(img, (20,20), interp='nearest') / 255.0
  img = imresize(img, (L,L)) / 255.0
  img = gaussian_filter(img, sigma=0.5)
  thold = 0.2
  img[img > thold] = 1
  img[img < thold] = 0
  return img, _x

class Knight:
  L = 6
  STATES = ["U", "L", "R"]
  ACTIONS = ["UP", "LEFT", "RIGHT"]

  def gen_s(self):
    L = self.L
    ret = np.random.randint(-L,L,size=2)
    # ret[1] = L
    if self.goal(ret): return self.gen_s()
    else: return ret
    

  def step(self, s, a):
    if a == "UP": return s - np.array([1,2]) 
    if a == "LEFT": return s - np.array([-2,-1]) 
    if a == "RIGHT": return s - np.array([2,-1]) 
    else: assert 0

  def goal(self, s):
    return s[0] == 0 and s[1] == 0


  def get_trace(self, actor, bound=10, s=None):
    trace = []
    s = gen_s() if s == None else s
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s)
      ss = self.step(s, action)
      reward = 1.0 if self.goal(ss) else move_reward
      trace.append((s,action,ss,reward))
      if self.goal(ss): return trace
      s = ss
    return trace

  def abstract(self, s):
    diffx, diffy = s[0], s[1]
    if diffy > 0: return "U"
    if diffx <= 0: return "L"
    else: return "R"

    return vertical + horizontal

class Flag:
  L = 5
  STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]
  ACTIONS = ["1++", "2--", "swp", "stop"]

  def gen_s(self):
    L = self.L
    pt1, pt2 = 0, L-1
    ret = pt1, pt2, np.random.randint(0,2,size=L)
    if self.goal(ret): return self.gen_s()
    else: return ret

  def goal(self, state):
    _,_,s = state
    flag = False
    for x in s:
      if x == 1:
        flag = True
      if x == 0 and flag:
        return False
    return True 

  def step(self, s, a):
    p1, p2, ary = s
    if a == "1++": return min(p1 + 1, self.L - 1), p2, ary
    if a == "2--": return p1, max(p2 - 1, 0), ary
    if a == "swp": 
      new_ary = np.copy(ary)
      new_ary[p1] = ary[p2]
      new_ary[p2] = ary[p1]
      return p1, p2, new_ary
    if a == "stop":
      return s
    else: assert 0

  def get_trace(self, actor, bound=12, s=None):
    trace = []
    s = self.gen_s() if s == None else s
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s)
      ss = self.step(s, action)

      reward = 1.0 if (self.goal(ss) and action == "stop") else move_reward
      trace.append((s,action,ss,reward))
      if action == "stop": return trace
      s = ss
    return trace

  def abstract(self, s):
    p1, p2, ary = s
    ret = ""
    ret += "1" if p1 == p2 else "0"
    ret += "1" if ary[p1] == 1 else "0"
    ret += "1" if ary[p2] == 1 else "0"
    return ret

# doubling the number in bit representation
class BitDouble:
  L = 5
  # the four pointers are ptr1, carry, output
  # the values of pointers can be 0 or 1
  # pt1 and output are pointing at index 0 in beginning while carry points at index 1
  # all pointers increments by 1 with the ++ command
  # there is a program counter which always increases but can be reset
  STATES = []
  ACTIONS = ["++", "c0", "c1", "o0", "o1"]

  def gen_s(self):
    L = self.L
    pt1, ptc, pto, PC = 0, 1, 0, 0
    a1 = list(np.random.randint(0,2,size=L-1)) + [0]
    ao = [0 for i in range(L)]
    return pt1, ptc, pto, a1, ao, PC

  def __init__(self):
    for p1_content in ["0", "1"]:
      for c_content in ["0", "1"]:
        for o_content in ["0", "1"]:
          for pc in ["0", "1"]:
            self.STATES.append(p1_content+c_content+o_content+pc)


  def goal(self, state):
    pt1, ptc, pto, a1, ao, PC = state
    def meow(aaa):
      ret = ""
      for a in aaa:
        ret = str(a) + ret
      return int(ret, 2)
    return meow(a1) + meow(a1) == meow(ao)

  def step(self, s, a):
    pt1, ptc, pto, a1, ao, PC = copy.deepcopy(s)
    if a == "++": return min(pt1 + 1, self.L - 1),\
                         min(ptc + 1, self.L - 1),\
                         min(pto + 1, self.L - 1), a1, ao, (PC+1) % 2
    if a == "c0": 
      ao[ptc] = 0
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
    if a == "c1": 
      ao[ptc] = 1
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
    if a == "o0": 
      ao[pto] = 0
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
    if a == "o1": 
      ao[pto] = 1
      return pt1, ptc, pto, a1, ao, (PC+1) % 2
 #   if a == "reset":
 #     return pt1, ptc, pto, a1, ao, 0
    assert 0

  def get_trace(self, actor, s=None):
    bound = self.L * 6
    def stop(s):
      pt1, ptc, pto, a1, ao, PC = copy.deepcopy(s)
      return ptc == pto

    trace = []
    s = self.gen_s() if s == None else s
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s)
      ss = self.step(s, action)

      reward = 1.0 if (stop(ss) and self.goal(ss)) else move_reward
      trace.append((s,action,ss,reward))
      if stop(ss): return trace
      s = ss
    return trace

  def abstract(self, s):
    pt1, ptc, pto, a1, ao, PC = copy.deepcopy(s)
    ret = ""
    ret += str(a1[pt1])
    ret += str(ao[ptc])
    ret += str(ao[pto])
    ret += str(PC)
    return ret

class BugZero:

  L = 16
  STATES = []
  ACTIONS = ['R', 'U', 'L', 'D']
  
  def gen_s(self):

    L = self.L
    ret = np.zeros([L, L]) 
    for i in range(10):
      img_l = max( L / 2 - i, 4)
      # print img_l
      digit = get_mnist_img(img_l)[0]
      x_displace = np.random.randint(0,L - img_l)
      y_displace = np.random.randint(0,L - img_l)
      digit = np.lib.pad(digit, ((x_displace, L - img_l - x_displace), 
                                 (y_displace, L - img_l - y_displace)), 'minimum')

      ret = ret + digit
    ret = np.clip(ret, 0, 1)

    def to_edge(kk, ll):
      if kk == 0: return (ll, 0)
      if kk == 1: return (L-1, ll)
      if kk == 2: return (ll, L-1)
      if kk == 3: return (0, ll)
      assert 0, "impossible!"

    start = np.random.randint(0,4), np.random.randint(0,L)
    end = (start[0] + 2) % 4, L - 1 - start[1]

    return ret, to_edge(*start), to_edge(*end)

  def goal(self, state):
    maze, start, end = state
    return start == end

  def step(self, s, a):
    L = self.L
    maze, start, end = s
    px, py = start
    if a == 'R': nxt = min(px + 1, L-1), py
    if a == 'U': nxt = px, max(py-1, 0)
    if a == 'L': nxt = max(px-1, 0), py
    if a == 'D': nxt = px, min(py+1, L-1)

    nx, ny = nxt
    if maze[ny][nx] == 1: nxt = start
    return maze, nxt, end
 
  def get_trace(self, actor, s=None):
    s = self.gen_s() if s == None else s
    bound = self.L * 4
    trace = []
    move_reward = -0.1
    for i in range(bound):
      action = actor.act(s, self)
      ss = self.step(s, action)

      reward = 1.0 if self.goal(ss) else move_reward
      trace.append((s,action,ss,reward))
      if self.goal(ss): return trace
      s = ss
    return trace

  def gen_a_star_actor(self, problem):
    a_star_sol = self.a_star_solution(problem)
    class AgentAStar:
      def __init__(self, a_star_sol):
        self.sol = a_star_sol
        # print a_star_sol
      def act(self, s, env):
        maze, start, end = s
        # print "cur state ", start
        start_idx = self.sol.index(start)
        nxt_idx = start_idx + 1
        nxt = self.sol[nxt_idx]

        nx, ny = nxt
        sx, sy = start
        if nx - sx == 1: 
          # print " R "
          return 'R'
        if nx - sx == -1: 
          # print " L "
          return 'L'
        if ny - sy == 1: 
          # print " D "
          return 'D'
        if ny - sy == -1: 
          # print " U "
          return 'U' 
    return AgentAStar(a_star_sol)

  # actually BFS because fuck it LOL
  def a_star_solution(self, s):

    def heuristic_cost(aa, bb):
      return 0
#       xx,yy = aa  
#       xxx, yyy = bb
#       return abs(xxx-xx) + abs(yyy - yy)

    maze, start, end = s
    L = self.L

    def get_neighbor(pos):
      px, py = pos
      neib = [(px, min(py+1,L-1)), (px, max(py-1,0)), 
              (max(px-1,0), py), (min(px+1,L-1), py)]

      # note the matrix is stored in Y X, i.e. the outer index is Y
      # thus we need to read the transposed version of it otherwise it's weird
      filt_neib = [nn for nn in neib if maze[nn[1]][nn[0]] == 0]
      
      return filt_neib

    fringe = [(0, start)]
    seen = set()
    back_ptr = dict()    

    # keeps track of the minimum distance from node to start
    min_dists = dict()

    while len(fringe) > 0:

      # print len(seen), len(fringe)
      # print fringe
      to_pop = min(fringe)
      fringe.remove(to_pop)
      
      cur_cost, cur_node = to_pop
      if cur_node == end: 
        break
      # add currently popped node to seen
      seen.add(cur_node)
      neib = get_neighbor(cur_node)
      neib = [nn for nn in neib if nn not in seen]

      for nn in neib: 
        back_ptr[nn] = cur_node
        # TODO WARNING: THIS MIGHT BE STUPID LOL (adding a small number)

        # probabilistically add a "fixed" small cost per nn node
        node_dist = cur_cost + 1000 + np.random.randint(0,10)
        if nn in min_dists:
          continue
        else:
          min_dists[nn] = node_dist

        fringe.append((node_dist, nn))
      fringe = list(set(fringe))

    ret = []
    cur_track = end
    while cur_track != start:
      ret.append(cur_track)
      cur_track = back_ptr[cur_track]
    ret.append(cur_track)

    assert len(ret) < 100, "YOUR SMALL NUMBER IS FUCKING U UP RIGHT NOW BOI"

    return list(reversed(ret))

  def vectorize_state(self, s):
    maze, start, end = s
    stacked = np.dstack( (maze, np.zeros((self.L,self.L)),
                                np.zeros((self.L,self.L))) )
    startx, starty = start
    endx, endy = end
    stacked[starty][startx][1] = 1
    stacked[endy][endx][2] = 1

    return stacked

    

class RandomActor:
  def __init__(self, actions):
    self.actions = actions

  def act(self, s):
    ret = self.actions[np.random.randint(len(self.actions))]
    return ret

def print_Q(Q):
  keys = sorted(list(Q.keys()))
  print keys
  for k in keys:
    print k, Q[eval(k)]


class Experience:
  
  def __init__(self, buf_len):
    self.buf = []
    self.buf_len = buf_len

  def trim(self):
    while len(self.buf) > self.buf_len:
      self.buf.pop()

  def add(self, trace):
    self.buf += trace
    self.trim()
  
  def sample(self):
    idxxs = np.random.choice(len(self.buf), size=1, replace=False)
    return self.buf[idxxs[0]]

  
