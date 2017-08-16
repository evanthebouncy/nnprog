import tensorflow as tf

def weight_variable(shape):
  print shape
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def get_architecture(num_filter, ker_size, num_layers):

  def makeModel(imageInput):
    cur = imageInput

    print cur
    channel_size = 3

    for _ in range(num_layers):

      W_conv = weight_variable([ker_size, ker_size, channel_size, channel_size])
      b_conv = bias_variable([channel_size])

      xxx_R17 = conv2d(cur, W_conv)
      print "xxxr17 ", xxx_R17
      cur = tf.nn.relu(xxx_R17 + b_conv)
  
    print cur
    return cur

  return makeModel

# 4 output channels, 2x2 kernel size, default stride of 1, and 8 times folded over 
# architecture1 = get_architecture(4, 2, 8)

