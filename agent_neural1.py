import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from auxiliary import *

class RandomAgentNeural(object):

    def __init__(self, initial_state, initial_action, step_size, sess, targets):
        initial_state = initial_state
        initial_state = initial_state.reshape((9,))
        self._step_size = step_size
        self._targets = targets

        self._old_action = tf.get_variable(initializer=initial_action, name='old_action')
        self._old_action_feed = tf.placeholder(tf.int32, [])
        self._old_action_update = self._old_action.assign(self._old_action_feed)

        self._new_action = tf.get_variable(initializer=initial_action, name='new_action')
        self._new_action_feed = tf.placeholder(tf.int32, [])
        self._new_action_update = self._new_action.assign(self._new_action_feed)

        self._x_old_even = tf.get_variable(initializer=initial_state, name='old_even_state')
        self._x_old_even_feed = tf.placeholder(tf.float32, [9,])
        self._x_old_even_update = self._x_old_even.assign(self._x_old_even_feed)

        self._x_old = tf.get_variable(initializer=initial_state, name='old_state')
        self._x_old_feed = tf.placeholder(tf.float32, [9,])
        self._x_old_update = self._x_old.assign(self._x_old_feed)

        self._x_new = tf.get_variable(initializer=initial_state, name='new_state')
        self._x_new_feed = tf.placeholder(tf.float32, [9,])
        self._x_new_update = self._x_new.assign(self._x_new_feed)

        self._y_target = tf.get_variable(initializer=np.zeros((1,1), dtype='float32'), name='target_value')
        self._y_target_feed = tf.placeholder(tf.float32, [1,1])
        self._y_target_update = self._y_target.assign(self._y_target_feed)

        initializer = tf.contrib.layers.xavier_initializer() #tf.zeros_initializer() #
        self._W_fc1 = tf.Variable(initializer(shape=[9,100]))
        self._b_fc1 = tf.Variable(initializer(shape=[100]))
        self._W_fc2 = tf.Variable(initializer(shape=[100,50]))
        self._b_fc2 = tf.Variable(initializer(shape=[50]))
        self._W_fc3 = tf.Variable(initializer(shape=[50,1]))
        self._b_fc3 = tf.Variable(initializer(shape=[1]))

        self._h_new1 = tf.nn.tanh(tf.matmul(tf.reshape(self._x_new, [1,9]), self._W_fc1) + self._b_fc1)
        self._h_new2 = tf.nn.tanh(tf.matmul(self._h_new1, self._W_fc2) + self._b_fc2)
        self._y_hat_new = tf.nn.tanh(tf.matmul(self._h_new2, self._W_fc3) + self._b_fc3)
        #self._y_hat_new = tf.reduce_sum(self._y_hat_new)
        #self._y_hat_new = tf.matmul(self._x_new, self._W_fc1) + self._b_fc1
        #self._y_hat_new = tf.reduce_mean(self._y_hat_new)

        self._h_old1 = tf.nn.tanh(tf.matmul(tf.reshape(self._x_old, [1,9]), self._W_fc1) + self._b_fc1)
        self._h_old2 = tf.nn.tanh(tf.matmul(self._h_old1, self._W_fc2) + self._b_fc2)
        self._y_hat_old = tf.nn.tanh(tf.matmul(self._h_old2, self._W_fc3) + self._b_fc3)
        #self._y_hat_old = tf.reduce_sum(self._y_hat_old)
        #self._y_hat_old = tf.matmul(self._x_old, self._W_fc1) + self._b_fc1
        #self._y_hat_old = tf.reduce_mean(self._y_hat_old)

        #self._squared_loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self._y_target, predictions=self._y_hat_old))
        self._squared_loss = tf.reduce_sum(tf.square(self._y_target - self._y_hat_old))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=step_size)
        self._train_step = optimizer.minimize(self._squared_loss)

        self._sess = sess

    def step(self, new_state, reward, discount):

        x = new_state.reshape((9,))
        ref_idx = x

        # Behaviour policy
        empty_spaces = np.where(x == 0)[0].astype('int32')
        action = np.random.choice(empty_spaces)
        self._sess.run(self._new_action_update, feed_dict={self._new_action_feed: action})

        q_idx = np.copy(ref_idx)
        q_idx[action] = 1
        q_idx_feed = q_idx
        self._sess.run(self._x_new_update, feed_dict={self._x_new_feed: q_idx_feed})

        x_old_even = self._x_old_even.eval()
        ref_idx_old = x_old_even

        q_idx_old = np.copy(ref_idx_old)
        action_old = self._old_action.eval()
        q_idx_old[action_old] = 1

        q_idx_old_feed = q_idx_old
        self._sess.run(self._x_old_update, feed_dict={self._x_old_feed: q_idx_old_feed})


        #q_new = self._y_hat_new.eval()
        q_idx_targets = q_idx + 1
        q_new = self._targets[tuple(q_idx_targets)]
        y_new = np.reshape(reward + discount * q_new, (1,1))
        self._sess.run(self._y_target_update, feed_dict={self._y_target_feed: y_new})
        # Neural network for learning parameter vector pertaining to q(s) linear transformation.
        self._sess.run(self._train_step) #feed_dict={self._y_target: y_new})

        # Store current state and action as old state and action.
        self._sess.run(self._old_action_update, feed_dict={self._old_action_feed: action})
        self._sess.run(self._x_old_even_update, feed_dict={self._x_old_even_feed: x})

        return action


class GreedyTestAgentNeural(object):
    def __init__(self, W1, b1, W2, b2, W3, b3, sess):

        self._x = tf.placeholder(tf.float32, [9,])
        self._W_fc1 = tf.constant(W1)
        self._b_fc1 = tf.constant(b1)
        self._W_fc2 = tf.constant(W2)
        self._b_fc2 = tf.constant(b2)
        self._W_fc3 = tf.constant(W3)
        self._b_fc3 = tf.constant(b3)

        #self._y_hat = tf.matmul(self._x, self._W_fc1) + self._b_fc1
        #self._y_hat = tf.reduce_mean(self._y_hat)
        self._h1 = tf.nn.tanh(tf.matmul(tf.reshape(self._x, [1,9]), self._W_fc1) + self._b_fc1)
        self._h2 = tf.nn.tanh(tf.matmul(self._h1, self._W_fc2) + self._b_fc2)
        self._y_hat = tf.nn.tanh(tf.matmul(self._h2, self._W_fc3) + self._b_fc3)
        #self._y_hat = tf.reduce_mean(self._y_hat)

        self._sess = sess

    def step(self, new_state):

        x = new_state.reshape((9,))
        ref_idx = x
        empty_spaces = np.where(x == 0)[0].astype('int32')

        q_comparison = np.zeros(len(empty_spaces))
        for a, i in enumerate(empty_spaces):
            q_comparison_idx = np.copy(ref_idx)
            q_comparison_idx[i] = 1 #2 # N.B. 2 is the q-index for a cross
            q_comparison[a] = self._sess.run(self._y_hat, feed_dict={self._x: q_comparison_idx})

        q_argmax = np.argwhere(q_comparison == np.amax(q_comparison))
        q_argmax = q_argmax[:, -1]
        if np.shape(q_argmax) == (1,):
            action = empty_spaces[q_argmax[0]]
        else:
            action = np.random.choice(empty_spaces[q_argmax])
        return action