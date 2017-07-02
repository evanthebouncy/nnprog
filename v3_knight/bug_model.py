from architecture import *
import numpy as np

# 4 output channels, 2x2 kernel size, default stride of 1, and 8 times folded over 
architecture = get_architecture(4, 2, 8)

def flattenImageOutput(c1):
  c1d = int(c1.shape[1]*c1.shape[2]*c1.shape[3])
  print "fully connected input dimensionality:",c1d
  f1 = tf.reshape(c1, [-1, c1d])
  return f1

class Oracle:

  def __init__(self, L):

    self.graph = tf.Graph()
    self.session = tf.Session(graph = self.graph)
  
    with self.session.graph.as_default():
      self.input_var = tf.placeholder(tf.float32, [None, L, L, 3])
      self.target_var = tf.placeholder(tf.int32, [None])
      self.image_representation = architecture(self.input_var)
      print self.image_representation
      self.image_flat = flattenImageOutput(self.image_representation)
      print self.image_flat

      self.image_flat = tf.nn.dropout(self.image_flat, 0.9)

      self.prediction = tf.layers.dense(self.image_flat, 4, activation = tf.nn.relu)

      self.pred_prob = tf.nn.softmax(self.prediction)

      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.target_var,
                                                                 logits = self.prediction)

      self.loss = tf.reduce_sum(self.loss)
      self.optimizer = tf.train.AdamOptimizer(0.001)
      self.train = self.optimizer.minimize(self.loss)
      initializer = tf.global_variables_initializer()
      self.session.run(initializer)

      self.saver = tf.train.Saver()


  def generate_feed(self, batch):
    targets, inputs = batch
    to_return = dict()
    to_return[self.input_var] = inputs
    to_return[self.target_var] = targets
    return to_return

  def generate_act_feed(self, vector_state):
    to_return = dict()
    to_return[self.input_var] = vector_state
    return to_return

  def train_model(self, batch):
    current_loss, _ = self.session.run([self.loss, self.train], self.generate_feed(batch))
    print current_loss

  def save_model(self, path):
    self.saver.save(self.session, path)
    print "model saved at ", path

  def restore_model(self, path):
    self.saver.restore(self.session, path)
    print "model restored  from ", path

  # only supports 1 state at a time, no batching plz
  def act(self, state, env):
    vector_state = env.vectorize_state(state)
    feed_dict = self.generate_act_feed(np.array([vector_state]))
    the_action = self.session.run([self.pred_prob], feed_dict)[0][0]
    print the_action
    move_idx = np.random.choice([0,1,2,3], p=the_action)
    return env.ACTIONS[move_idx]
    







