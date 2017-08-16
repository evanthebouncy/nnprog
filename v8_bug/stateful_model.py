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

class StatelessAgent:

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

      # use to supervise pain
      self.true_pain = tf.placeholder(tf.float32, [None, 2])

      # one layer of fc to predict the action
      self.image_flat = tf.layers.dense(self.input_state, 200, activation= tf.nn.relu)

      # add a dropout to regularize
      self.image_flat = tf.layers.dropout(self.image_flat, rate=0.1)

      # predict the move
      self.prediction = tf.layers.dense(self.image_flat, action_dim)
      self.pred_prob = tf.nn.softmax(self.prediction)

      # predict the pain
      self.pain_flat = tf.layers.dense(self.input_state, 200, activation= tf.nn.relu)
      self.pain_prediction = tf.layers.dense(self.pain_flat, 2)
      self.pain_prob = tf.nn.softmax(self.pain_prediction)

      self.pred_prob = self.pred_prob + 1e-8
      self.pain_prob = self.pain_prob + 1e-8

      # set up the cost function for training
      self.log_pred_prob = tf.log(self.pred_prob)
      self.objective = tf.reduce_mean(self.log_pred_prob * self.roll_out_reward)

      # set up the cost function for pain training
      self.pain_loss = tf.reduce_mean(-tf.reduce_sum(self.true_pain * tf.log(self.pain_prob), reduction_indices=[1]))      
      
      # add regularization to prevent blow-up
      all_vars = tf.trainable_variables()
      squarez = [tf.reduce_sum(tf.square(vv)) for vv in all_vars]
      self.regularizer = sum(squarez)
    
      self.loss = -self.objective

      self.optimizer = tf.train.AdamOptimizer(0.001)
      self.train = self.optimizer.minimize(self.loss)
      self.pain_train = self.optimizer.minimize(self.pain_loss)

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
    batch_pains = []
    for trace in trace_batch:
      states = [self.xform(tr[0]) for tr in trace]
      actions = [tr[1] for tr in trace]
      disc_rewards = get_discount_future_reward(trace) 
      for s, name_a_p, r in zip(states, actions, disc_rewards):
        # print s, name_a, r
        name, a, pain = name_a_p
        batch_states.append(s)
        batch_action_indexed_rewards.append(r * self.xform_action(a))
        if trace[-1][-1] == 1.0:
          batch_pains.append(np.array([0.0, 1.0]))
        else:
          batch_pains.append(np.array([1.0, 0.0]))

    if batch_states == []: return

    batch_states = np.array(batch_states)
    batch_action_indexed_rewards = np.array(batch_action_indexed_rewards)
    batch_pains = np.array(batch_pains)

    if to_learn == "move":
      self.session.run([self.train], {self.input_state: batch_states,
                                      self.roll_out_reward: batch_action_indexed_rewards})
    if to_learn == "pain":
      self.session.run([self.pain_train], {self.input_state: batch_states,
                                           self.true_pain: batch_pains})

  # only supports 1 state at a time, no batching plz
  def act(self, state):
    blah1, blah2, blah3, blah4, call_st = state
    xform_state = self.xform(state)
    inp = np.array([xform_state])
    the_action = self.session.run([self.pred_prob], {self.input_state: inp})[0][0]
    if self.inspect: print env.abstract(state), the_action, self.session.run([self.prediction], {self.input_state: inp})[0][0]
    move_idx = np.random.choice(range(self.action_dim), p=the_action)

    xform_state_fake = self.xform(state, [1.0, 0.0])
    inp_fake = np.array([xform_state_fake])
    blue_pain = self.session.run([self.pain_prob], {self.input_state: inp_fake})[0][0]

    return "bug", self.actions[move_idx], blue_pain
    
