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
    colr = ["b", "m"]
    dsp = [-0.1, 0.1]
    flip_idx = 0
    certainty = 1.0
    for idx, node in enumerate(path):
      nx, ny = node
      if action_path != None:
        nname, aa, pain = action_path[idx]
        if nname == "blu": flip_idx = 0
        if nname == "meg": flip_idx = 1
        # certainty = 1.0 - pain
        certainty = 1.0
      plt.scatter(x=[nx+dsp[flip_idx]], y=[ny], c=colr[flip_idx], s=40, alpha=certainty)

  # draw in the start and end last so the blue path don't hide them
  grn_x, grn_y, = start
  red_x, red_y = end
  plt.scatter(x=[grn_x], y=[grn_y], c='g', s=40)
  plt.scatter(x=[red_x], y=[red_y], c='r', s=40)

  plt.savefig(name)


