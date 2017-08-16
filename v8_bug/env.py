import numpy as np
from numpy import array
from scipy.misc import imresize
import copy
from scipy.ndimage.filters import gaussian_filter
import random


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

class BugZero:

  STATES = []
  ACTIONS = ['R', 'U', 'L', 'D']

  def diff(self, pos1, pos2):
    return pos2[0] - pos1[0], pos2[1] - pos1[1]

  def norm(self, thing):
    x, y = thing
    summ = abs(x) + abs(y) + 1e-8
    return float(x) / summ, float(y) / summ

  def __init__(self, L):
    self.L = L

  def valid_maze(self, s):
    maze, start, end, prev_path, info = s
    if start == end: return False
    sx, sy = start
    ex, ey = end
    if maze[sy][sx] == 1: return False
    if maze[ey][ex] == 1: return False
    return self.a_star_solution(s) != []
  
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

    # start = np.random.randint(0,4), np.random.randint(0,L)
    # end = (start[0] + 2) % 4, L - 1 - start[1]

    # return ret, to_edge(*start), to_edge(*end)
    s_start = (np.random.randint(0,L), np.random.randint(0,L))
    s_goal = (np.random.randint(0,L), np.random.randint(0,L))
    toret = ret, s_start, s_goal, [], {"s_start":s_start}
    if self.valid_maze(toret): return toret
    else: return self.gen_s() 

  def goal(self, state):
    maze, start, end, prev_path, info = state
    return start == end

  def step(self, s, a):
    maze, start, end, prev_path, info = s
    L = len(maze)
    px, py = start
    if a == 'R': nxt = min(px + 1, L-1), py
    if a == 'U': nxt = px, max(py-1, 0)
    if a == 'L': nxt = max(px-1, 0), py
    if a == 'D': nxt = px, min(py+1, L-1)

    nx, ny = nxt
    if maze[ny][nx] == 1: nxt = start
    return maze, nxt, end, prev_path + [(start, a)], info
 
  def get_trace(self, actor, s=None, test=False):
    s = self.gen_s() if s == None else s
    maze, start, end, prev_path, info = s
    L = len(maze)
    bound = L * 8
    trace = []
    move_reward = -0.001
    for i in range(bound):
      name, action, blue_pain = actor.act(s)

      maze, start, end, prev_path, info = s

      ss = self.step(s, action)

      reward = 1.0 if self.goal(ss) else move_reward
      trace.append((s,(name,action, blue_pain),ss,reward))
      if self.goal(ss): return trace
      s = ss
    return trace

  def trace_to_path(self, trace):
    return [tr[0][1] for tr in trace]

  def trace_to_action_path(self, trace):
    return [(tr[1][0], tr[1][1], tr[1][2][0]) for tr in trace]

  def gen_a_star_actor(self, problem):
    a_star_sol = self.a_star_solution(problem)
    class AgentAStar:
      def __init__(self, a_star_sol):
        self.sol = a_star_sol
        # print a_star_sol
      def act(self, s):
        maze, start, end, prev_path, info = s
        # print "cur state ", start
        start_idx = self.sol.index(start)
        nxt_idx = start_idx + 1
        nxt = self.sol[nxt_idx]

        nx, ny = nxt
        sx, sy = start
        if nx - sx == 1: 
          # print " R "
          return "A*", 'R', 0.0
        if nx - sx == -1: 
          # print " L "
          return "A*", 'L', 0.0
        if ny - sy == 1: 
          # print " D "
          return "A*", 'D', 0.0
        if ny - sy == -1: 
          # print " U "
          return "A*", 'U', 0.0
    return AgentAStar(a_star_sol)

  # actually BFS because fuck it LOL
  def a_star_solution(self, s):

    def heuristic_cost(aa, bb):
      return 0
#       xx,yy = aa  
#       xxx, yyy = bb
#       return abs(xxx-xx) + abs(yyy - yy)

    maze, start, end, prev_path, info = s
    L = len(maze)

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

    if cur_track not in back_ptr:
      return []

    while cur_track != start:
      ret.append(cur_track)
      cur_track = back_ptr[cur_track]
    ret.append(cur_track)

    assert len(ret) < 100, "YOUR SMALL NUMBER IS FUCKING U UP RIGHT NOW BOI"

    return list(reversed(ret))

  def get_glimpse(self, s, window_size):
    maze, start, end, path, info = s
    L = (len(maze) - 1) / 2
    # assert the map is centered
    assert start == (L,L)
    # print maze[L,L], maze[L-1, L-1]
    ret = maze[L-window_size:L+window_size+1, 
               L-window_size:L+window_size+1]
    return ret

  def get_scent(self, s, window_size):
    maze, start, end, path, info = s
    L = (len(maze) - 1) / 2
    # assert the map is centered
    assert start == (L,L)
    # print maze[L,L], maze[L-1, L-1]
    scentmaze = np.zeros(shape=maze.shape)
    # print path
    for t, pp in enumerate(reversed(path)):
      poop_location, _ = pp
      ppx, ppy = poop_location
      scentmaze[ppx, ppy] += 2 ** -t

    # print scentmaze
    ret = scentmaze[L-window_size:L+window_size+1, 
               L-window_size:L+window_size+1]
    return ret
    
  def get_goal_direction(self, s):
    maze, start, end, prev_path, info = s
    s_x, s_y = start
    e_x, e_y = end
    dx, dy = e_x - s_x, e_y - s_y
    ll = abs(dx) + abs(dy)
    return float(dx) / ll, float(dy) / ll

  def get_start_direction(self, s):
    maze, start, end, prev_path, info = s
    s_x, s_y = start
    e_x, e_y = info["s_start"]
    dx, dy = e_x - s_x, e_y - s_y
    ll = abs(dx) + abs(dy)
    return float(dx) / ll, float(dy) / ll

  def vectorize_state(self, s):
    maze, start, end, path, info = s

    mazeL = maze.shape[0]

    stacked = np.dstack( (maze, np.zeros((mazeL,mazeL)),
                                np.zeros((mazeL,mazeL))) )
    startx, starty = start
    endx, endy = end
    stacked[starty][startx][1] = 1
    stacked[endy][endx][2] = 1

    return stacked

  def centered_state(self, s):
    maze, start, end, prev_path, info = s
    L = len(maze)

    bigmaze=np.zeros((2*L+1,2*L+1),maze.dtype) 
    x, y = start
    ex, ey = end
    bigmaze[L-y:2*L-y, L-x:2*L-x] = maze

    delta_x, delta_y = L-x, L-y
    delta_path = []
    for p in prev_path:
      pos, move = p
      pos_x, pos_y = pos
      delta_path.append(((pos_x + delta_x, pos_y + delta_y), move))

    return bigmaze, (L,L), (L+ex-x, L+ey-y), delta_path, info

  # takes in a path and produce a finite summary of it
  def fractal_memory(self, path):
    detail_steps = 4
    def get_indexs(d_steps):
      ret = []
      so_far = 0
      for i in range(0, 4):
        for j in range(d_steps) :
          ret.append(so_far)
          so_far += 2 ** i
      return ret

    idxxs = get_indexs(detail_steps)
    memory = []
    rev_path = list(reversed(path))
    for idxx in idxxs:
      if idxx < len(rev_path):
        memory.append(rev_path[idxx])
      else:
        memory.append(None)

    return memory

  # takes in a path and produce a finite summary of it
  # assume the path can never be empty
#   def fractal_memory(self, path):
#     rev_path = list(reversed(path))
#     memory = []
#     memory.append(rev_path[-1])
#     for i in range(1,8):
#       memory.append(rev_path[len(rev_path) / 2**i])  
#     return memory
      
  def fractal_path_state(self, s):
    maze, start, end, prev_path, info = s
    L = len(maze)
    
    pathh = [p[0] for p in prev_path]+[start]
    print "passed in ", pathh
    frac_mem = self.fractal_memory(pathh)
    print "what I got "
    print frac_mem
    frac_diffs = []
    for i in range(1,len(frac_mem)):
      if frac_mem[i] != None:
        difx, dify = self.diff(frac_mem[i], frac_mem[i-1])
        norxx, noryy = self.norm((difx, dify))
        frac_diffs += [norxx, noryy]
      else: 
        frac_diffs += [0.0, 0.0]
    
    first_state = prev_path[0][0] if len(prev_path) > 0 else start
    dx_start, dy_start = self.diff(first_state, start)
    
    dir_start = self.norm((dx_start, dy_start))
      
    print frac_diffs
       

    

class RandomActor:
  def __init__(self, actions):
    self.actions = actions

  def act(self, s, env):
    ret = self.actions[np.random.randint(len(self.actions))]
    return ("rand", ret)

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
      self.buf.pop(0)

  def add(self, trace):
    self.buf.append( trace )
    self.trim()

  def add_success(self, trace):
    if trace[-1][-1] == 1.0:
      self.buf.append(trace)
    

  def add_balanced(self, trace):
    #print self.check_success(), " before "
    succ = trace[-1][-1] == 1.0
    def find_first(values):
      for idx, tr in enumerate(self.buf):
        if tr[-1][-1] in values: return idx
      assert 0

    if succ: 
      self.buf.pop(find_first([1.0]))
      self.buf.append(trace)

    else:
      self.buf.pop(find_first([0.001, -0.001, -0.002]))
      self.buf.append(trace)
    #print self.check_success(), " aftr "
  
  def sample(self):
    idxxs = np.random.choice(len(self.buf), size=1, replace=False)
    return self.buf[idxxs[0]]

  def sample_transition(self):
    idxxs = np.random.choice(len(self.buf), size=1, replace=False)
    trace = self.buf[idxxs[0]]
    tr_idx = np.random.choice(len(trace), size=1, replace=False)
    return trace[tr_idx[0]]
    

  def n_sample(self, n):
    return [self.sample() for _ in range(n)]

  def n_sample_transition(self, n):
    return [self.sample_transition() for _ in range(n)]

  def check_success(self):
    if len(self.buf) == 0: return 0
    # print self.buf[0][-1][-1]
    return sum([1.0 for tr in self.buf if tr[-1][-1] == 1.0 ])

def get_accuracy(agent, env, n):
  acc = 0.0
  for jj in range(n):
    tr = env.get_trace(agent)
    if tr[-1][-1] == 1.0: acc += 1.0
  return acc / n

def get_accuracy_from_set(agent, env, test_set):
  n = len(test_set)
  acc = 0.0
  for i, tt in enumerate(test_set):
    tr = env.get_trace(agent, s=tt)
    if tr[-1][-1] == 1.0: acc += 1.0
  return acc / n

def populate_experience(environment, agent, experience, exp_size):
  while experience.check_success() < exp_size / 2:
    maze = environment.gen_s()
    tr = environment.get_trace(agent, maze)
    experience.add_success(tr)
  for _ in range(exp_size / 2):
    maze = environment.gen_s()
    tr = environment.get_trace(agent, maze)
    experience.add(tr)

from numpy import array
def find_collapse_state(transition_batch, xform):
  act_map = dict()
  for tr in transition_batch:
    s, n_a, ss, r = tr
    s = xform(s)
    name, act, pain = n_a
    if act not in act_map:
      act_map[act] = set()
    act_map[act].add(repr(s))
  statez, other_statez = [], []
  for a in act_map:
    similar_states = list(act_map[a])
    for i in range(len(similar_states) / 2):
      statez.append(eval(similar_states[2*i]))
      other_statez.append(eval(similar_states[2*i + 1]))
  return statez, other_statez
