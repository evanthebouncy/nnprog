import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp

from matplotlib import figure

FIG = plt.figure()


def draw(maze, name, path=None):
  FIG.clf()

  matrix, start, end, prev_path = maze

  matrix = np.copy(matrix)
  ax = FIG.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
  # plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
  plt.colorbar()



  if path != None:
    for node in path:
      nx, ny = node
      plt.scatter(x=[nx], y=[ny], c="b", s=40)

  # draw in the start and end last so the blue path don't hide them
  grn_x, grn_y, = start
  red_x, red_y = end
  plt.scatter(x=[grn_x], y=[grn_y], c='g', s=40)
  plt.scatter(x=[red_x], y=[red_y], c='r', s=40)

  plt.savefig(name)


