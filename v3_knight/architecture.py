import tensorflow as tf

def get_architecture(num_filter, ker_size, num_layers):

  def makeModel(imageInput):
    cur = imageInput
    for _ in range(num_layers):
      cur = tf.layers.conv2d(inputs = cur,
                             filters = num_filter,
                             kernel_size = ker_size, 
                             activation = tf.nn.relu)
    return cur

  return makeModel

# 4 output channels, 2x2 kernel size, default stride of 1, and 8 times folded over 
# architecture1 = get_architecture(4, 2, 8)

