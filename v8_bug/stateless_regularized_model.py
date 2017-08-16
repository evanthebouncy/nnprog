import numpy as np
import tensorflow as tf

def get_discount_future_reward(trace):
  rewards = [tr[3] for tr in trace]
  disc_rewards = [0 for _ in range(len(rewards))]
  running_add = 0
  for t in reversed(range(0, len(rewards))):
    running_add = running_add * 0.98 + rewards[t]
    disc_rewards[t] = running_add
  return disc_rewards


def weight_variable(shape, w=0.1):
  initial = tf.truncated_normal(shape, stddev=w)
  return tf.Variable(initial)

def bias_variable(shape, w=0.1):
  initial = tf.constant(w, shape=shape)
  return tf.Variable(initial)

class StatelessAgentReg:

  inspect = False

  # pass in a xform function to transform the state into the vectors
  def __init__(self, name, state_dim, action_dim, xform, xform_action, actions):
    self.name = name
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.graph = tf.Graph()
    self.session = tf.Session(graph = self.graph)
    self.xform = xform
    self.xform_action = xform_action
    self.actions = actions
  
    with self.session.graph.as_default():
      # this is the input state
      self.input_state = tf.placeholder(tf.float32, [None, state_dim])
      # this is the roll-out-reward indexed by action on that particular state
      # used for training only
      self.roll_out_reward = tf.placeholder(tf.float32, [None, action_dim])

      self.input_state_other = tf.placeholder(tf.float32, [None, state_dim])

      # one layer of fc to predict the action
      W_embed = weight_variable([state_dim, 200])
      b_embed = bias_variable([200])

      # two embeddings
      self.state_embedding = tf.nn.sigmoid(tf.matmul(self.input_state, W_embed) + b_embed)
      self.state_embedding_other = tf.nn.sigmoid(tf.matmul(self.input_state_other, W_embed) + b_embed)

      # predict the move
      self.prediction = tf.layers.dense(self.state_embedding, action_dim)
      self.pred_prob = tf.nn.softmax(self.prediction)
      self.pred_prob = self.pred_prob + 1e-8

      # set up the cost function for training
      self.log_pred_prob = tf.log(self.pred_prob)
      self.objective = tf.reduce_mean(self.log_pred_prob * self.roll_out_reward)
    
      self.loss = -self.objective

      self.optimizer = tf.train.AdamOptimizer(0.001)
      self.train = self.optimizer.minimize(self.loss)

      # put a regularization cost on the state embedding
      state_diff = tf.square(self.state_embedding - self.state_embedding_other)
      self.reg_cost = tf.reduce_mean(state_diff)
      self.reg_train = self.optimizer.minimize(self.reg_cost)

      initializer = tf.global_variables_initializer()
      self.session.run(initializer)

      self.saver = tf.train.Saver()

  def save_model(self, path):
    self.saver.save(self.session, path)
    print "model saved at ", path

  def restore_model(self, path):
    self.saver.restore(self.session, path)
    print "model restored  from ", path

  def learn(self, trace_batch, to_learn):
    batch_states = []
    batch_action_indexed_rewards = []
    for trace in trace_batch:
      states = [self.xform(tr[0]) for tr in trace]
      actions = [tr[1] for tr in trace]
      disc_rewards = get_discount_future_reward(trace) 
      for s, name_a_p, r in zip(states, actions, disc_rewards):
        # print s, name_a, r
        name, a, pain = name_a_p
        batch_states.append(s)
        batch_action_indexed_rewards.append(r * self.xform_action(a))

    if batch_states == []: return

    batch_states = np.array(batch_states)
    batch_action_indexed_rewards = np.array(batch_action_indexed_rewards)

    if to_learn == "move":
      self.session.run([self.train], {self.input_state: batch_states,
                                      self.roll_out_reward: batch_action_indexed_rewards})

  def regularize(self, states, other_states):
    # states = [self.xform(ss) for ss in states]
    # other_states = [self.xform(ss_o) for ss_o in other_states]
    batch_states = np.array(states)
    batch_other_states = np.array(other_states)

    self.session.run([self.reg_cost, self.reg_train], {self.input_state: batch_states,
                                  self.input_state_other: batch_other_states})

  # only supports 1 state at a time, no batching plz
  def act(self, state):
    blah1, blah2, blah3, blah4, call_st = state
    xform_state = self.xform(state)
    inp = np.array([xform_state])
    the_action = self.session.run([self.pred_prob], {self.input_state: inp})[0][0]
    if self.inspect: print env.abstract(state), the_action, self.session.run([self.prediction], {self.input_state: inp})[0][0]
    move_idx = np.random.choice(range(self.action_dim), p=the_action)

    xform_state_fake = self.xform(state)
    inp_fake = np.array([xform_state_fake])

    return "bug", self.actions[move_idx], [0.0, 0.0]
    
