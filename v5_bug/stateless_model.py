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

      # one layer of fc to predict the action
      self.image_flat = tf.layers.dense(self.input_state, 200, activation= tf.nn.relu)

      # add a dropout to regularize
      self.image_flat = tf.layers.dropout(self.image_flat, rate=0.1)

      self.prediction = tf.layers.dense(self.image_flat, action_dim)
      self.pred_prob = tf.nn.softmax(self.prediction)

      # small_constant = tf.constant(1e-4, shape=self.pred_prob.get_shape(), name="small_delta")
      self.pred_prob = self.pred_prob + 1e-8

      # set up the cost function for training
      self.log_pred_prob = tf.log(self.pred_prob)
      self.objective = tf.reduce_mean(self.log_pred_prob * self.roll_out_reward)
      
      # add regularization to prevent blow-up
      all_vars = tf.trainable_variables()
      squarez = [tf.reduce_sum(tf.square(vv)) for vv in all_vars]
      self.regularizer = sum(squarez)
    
      self.loss = -self.objective

      self.optimizer = tf.train.AdamOptimizer(0.001)
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

  def learn(self, trace_batch):
    batch_states = []
    batch_action_indexed_rewards = []
    for trace in trace_batch:
      states = [self.xform(tr[0]) for tr in trace]
      actions = [tr[1] for tr in trace]
      disc_rewards = get_discount_future_reward(trace) 
      for s, name_a, r in zip(states, actions, disc_rewards):
        # print s, name_a, r
        name, a = name_a
        batch_states.append(s)
        batch_action_indexed_rewards.append(r * self.xform_action(a))

    if batch_states == []: return

    batch_states = np.array(batch_states)
    batch_action_indexed_rewards = np.array(batch_action_indexed_rewards)

    # print "regulairzation "
    # print self.session.run([self.regularizer], {self.input_state: batch_states,
    #                                 self.roll_out_reward: batch_action_indexed_rewards})

    # print batch_states.shape, batch_action_indexed_rewards.shape
    # print batch_states[0], batch_action_indexed_rewards[0]
    self.session.run([self.train], {self.input_state: batch_states,
                                    self.roll_out_reward: batch_action_indexed_rewards})

  # only supports 1 state at a time, no batching plz
  def act(self, state):
    xform_state = self.xform(state)
    inp = np.array([xform_state])
    the_action = self.session.run([self.pred_prob], {self.input_state: inp})[0][0]
    if self.inspect: print env.abstract(state), the_action, self.session.run([self.prediction], {self.input_state: inp})[0][0]
    move_idx = np.random.choice(range(self.action_dim), p=the_action)
    return self.name, self.actions[move_idx]
    
