import numpy as np
import tensorflow as tf

class Discriminator:

  # pass in a xform function to transform the state into the vectors
  def __init__(self, name, state_dim, action_dim, xform, xform_action):
    self.name = name
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.graph = tf.Graph()
    self.session = tf.Session(graph = self.graph)
    self.xform = xform
    self.xform_action = xform_action
  
    with self.session.graph.as_default():
      # this is the input state
      self.input_state = tf.placeholder(tf.float32, [None, state_dim])
      # this is the input action
      self.input_action = tf.placeholder(tf.float32, [None, action_dim])
      # this is the label of true-false for the classifier
      self.prediction_label = tf.placeholder(tf.float32, [None, 2])

      self.s_a = tf.concat([self.input_state, self.input_action], axis=1)

      # one layer of fc to predict the action
      self.hidden_state = tf.layers.dense(self.s_a, 100, activation= tf.nn.relu)

      # predict the move
      self.prediction = tf.layers.dense(self.hidden_state, 2)
      self.pred_prob = tf.nn.softmax(self.prediction)

      # add a small number so it doesn't blow up (logp or in action selection)
      self.pred_prob = self.pred_prob + 1e-8

      # set up the cost function for training
      self.log_pred_prob = tf.log(self.pred_prob)
      self.objective = tf.reduce_mean(self.log_pred_prob * self.prediction_label)

      self.loss = -self.objective

      self.optimizer = tf.train.AdamOptimizer(0.05)
      self.train = self.optimizer.minimize(self.loss)

      initializer = tf.global_variables_initializer()
      self.session.run(initializer)

      self.saver = tf.train.Saver()

  def save_model(self, path):
    self.saver.save(self.session, path)
    print "model saved at ", path

  def restore_model(self, path):
    self.saver.restore(self.session, path)
    print "model restored  from ", path

  def train_discrim(self, expert_sas, agent_sas):
    batch_states = []
    batch_actions = []
    batch_labels = []
    def add_sas(sas, lab):
      for s,a in sas:
        batch_states.append(self.xform(s))
        batch_actions.append(self.xform_action(a))
        batch_labels.append(lab)
    add_sas(expert_sas, np.array([0.0, 1.0]))
    add_sas(agent_sas, np.array([1.0, 0.0]))

    batch_states = np.array(batch_states)
    batch_actions = np.array(batch_actions)
    batch_labels = np.array(batch_labels)

    loss_train = self.session.run([self.loss, self.train], {self.input_state: batch_states,
                                    self.input_action: batch_actions,
                                    self.prediction_label: batch_labels})
    print "discrim loss ", loss_train[0]

  # only supports 1 state at a time, no batching plz
  def score(self, state, action):
    xform_state = self.xform(state)
    inp = np.array([xform_state])
    inp_action = np.array([self.xform_action(action)])
    the_score = self.session.run([self.pred_prob], {self.input_state: inp,\
                                                    self.input_action: inp_action})[0][0][1]
    return the_score
    
