import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp

from matplotlib import figure

FIG = plt.figure()


def draw(maze, name, path=None, action_path=None):
  FIG.clf()

  matrix, start, end, prev_path, call_state = maze

  matrix = np.copy(matrix)
  ax = FIG.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
  # plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
  plt.colorbar()



  if path != None:
    colr = "b"
    dsp = 0.0
    for idx, node in enumerate(path):
      nx, ny = node
      if action_path != None:
        nname, aa = action_path[idx]
        if aa == 'f0': 
          colr = 'b'
          dsp = -0.1
        if aa == 'f1': 
          colr = 'm'
          dsp = 0.1
      plt.scatter(x=[nx+dsp], y=[ny], c=colr, s=40)

  # draw in the start and end last so the blue path don't hide them
  grn_x, grn_y, = start
  red_x, red_y = end
  plt.scatter(x=[grn_x], y=[grn_y], c='g', s=40)
  plt.scatter(x=[red_x], y=[red_y], c='r', s=40)

  plt.savefig(name)


