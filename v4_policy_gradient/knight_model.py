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

class Driver:

  def __init__(self):

    self.graph = tf.Graph()
    self.session = tf.Session(graph = self.graph)
  
    with self.session.graph.as_default():
      # this is the input state
      self.input_state = tf.placeholder(tf.float32, [None, 3])
      # this is the roll-out-reward indexed by action on that particular state
      # used for training only
      self.roll_out_reward = tf.placeholder(tf.float32, [None, 3])

      # one layer of fc to predict the action
      self.image_flat = tf.layers.dense(self.input_state, 10, activation= tf.nn.relu)
      self.prediction = tf.layers.dense(self.image_flat, 3)
      self.pred_prob = tf.nn.softmax(self.prediction)

      # set up the cost function for training
      self.log_pred_prob = tf.log(self.pred_prob)
      self.objective = tf.reduce_mean(self.log_pred_prob * self.roll_out_reward)
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

  def learn(self, trace_batch, env):
    batch_states = []
    batch_action_indexed_rewards = []
    for trace in trace_batch:
      states = [env.abstract(tr[0]) for tr in trace]
      actions = [tr[1] for tr in trace]
      disc_rewards = get_discount_future_reward(trace) 
      for s, a, r in zip(states, actions, disc_rewards):
        batch_states.append(env.vectorize_abs_state(s))
        batch_action_indexed_rewards.append(r * env.vectorize_action(a))

    batch_states = np.array(batch_states)
    batch_action_indexed_rewards = np.array(batch_action_indexed_rewards)

    print batch_states[0], batch_action_indexed_rewards[0]
    self.session.run([self.train], {self.input_state: batch_states,
                                    self.roll_out_reward: batch_action_indexed_rewards})

  # only supports 1 state at a time, no batching plz
  def act(self, state, env):
    ab_state_id = env.STATES.index(env.abstract(state))
    inp = np.zeros([1, len(env.STATES)], np.float32)
    inp[0][ab_state_id] = 1.0
    the_action = self.session.run([self.pred_prob], {self.input_state: inp})[0][0]
    print env.abstract(state), the_action, self.session.run([self.prediction], {self.input_state: inp})[0][0]
    move_idx = np.random.choice([0,1,2], p=the_action)
    return env.ACTIONS[move_idx]
    
