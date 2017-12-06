from supervise import *
import skimage.measure
from draw import *

def dim_reduce(img, times):
  def _dim_r(img_in):
    return skimage.measure.block_reduce(img_in, (2,2), np.max)
  for t in range(times):
    img = _dim_r(img)
  return img

def enhance(img, x, y):
  return img[x*8:(x+1)*8, y*8:(y+1)*8]

def get_pyramid(img):
  return [dim_reduce(img, i) for i in range(4)]
  

if __name__ == "__main__":
  sas = sample_planner_sa(1)
  for i, s_a in enumerate(sas):
    s,a = s_a
    # print s[0]
    pyramid = get_pyramid(s[0])
    for j, p in enumerate(pyramid):
      draw(p, "drawings/ha{}_{}.png".format(i,j))
    for x in range(10):
      for y in range(10):
        enh = enhance(s[0], x, y)
        draw(enh, "drawings/enh_{}_{}_{}.png".format(i, x, y))
        

