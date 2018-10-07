import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from auxiliary import *

class RandomAgentNeural2(object):
    def __init__(self, initial_state, initial_action, step_size):
        # start state (2D state)
        self._s = initial_state
        #self._number_of_states = len(np.where(self._s == 0)[0])

        #x = np.reshape(self._initial_state, -1)
        #empty_spaces = np.where(x == 0)[0].astype('int32')
        self._action = initial_action
        self._step_size = step_size

        # Initialise parameter vector for linear transformation of q(s) for all actions.
        self._W1 = np.random.randn(9, 200)
        self._W2 = np.random.randn(200, 1)
        self._counter1 = 0
        self._counter2 = 0
    def q_values(self, state):
        h = relu(np.dot(state, self._W1))
        y = sigmoid(np.dot(h, self._W2))
        return y

    def step(self, new_state, reward, discount):
        action_old = self._action
        state_old = self._s

        x = np.reshape(new_state, -1)
        ref_idx = x #+ 1
        empty_spaces = np.where(x == 0)[0].astype('int32')
        '''
        # Target policy
        q_comparison = np.zeros(len(empty_spaces))
        for a, i in enumerate(empty_spaces):
            q_comparison_idx = np.copy(ref_idx)
            q_comparison_idx[i] = 2 # N.B. 2 is the q-index for a cross
            q_comparison[a] = self._q[tuple(q_comparison_idx)]

        q_argmax = np.argmax(q_comparison)
        action_max = empty_spaces[q_argmax]
        q_idx = np.copy(ref_idx)
        q_idx[action_max] = 2
        '''
        # Behaviour policy
        action = np.random.choice(empty_spaces)
        q_idx = np.copy(ref_idx)
        q_idx[action] = 1 #2


        x_old = np.reshape(state_old, -1)
        ref_idx_old = x_old #+ 1

        q_idx_old = np.copy(ref_idx_old)
        q_idx_old[action_old] = 1 #2

        q_idx = (q_idx - np.min(q_idx))/(np.max(q_idx) - np.min(q_idx))
        q_idx_old = (q_idx_old - np.min(q_idx_old)) / (np.max(q_idx_old) - np.min(q_idx_old))
        # 2. Update parameter vector pertaining to q(s) linear transformation.

        grad_2_part_1 = reward + discount * self.q_values(q_idx) - self.q_values(q_idx_old)
        grad_2_part_2 = d_sigmoid(np.dot(relu(np.dot(q_idx_old, self._W1)), self._W2))
        grad_2_part_3 = relu(np.dot(q_idx_old, self._W1))[np.newaxis, :]
        grad_2 =   grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2[:, np.newaxis])
        if np.sum(grad_2) == 0:
            self._counter1 += 1
        self._W2 = self._W2 + self._step_size * grad_2

        grad_1_part_1 = (grad_2_part_1*grad_2_part_2[:, np.newaxis]).dot(self._W2.T)
        grad_1_part_2 = d_relu(np.dot(q_idx_old, self._W1))
        grad_1_part_3 = q_idx_old[np.newaxis, :]
        grad_1 =   grad_1_part_3.T.dot(grad_1_part_1*grad_1_part_2)
        if np.sum(grad_1) == 0:
            self._counter2 += 1
        self._W1 = self._W1 + self._step_size * grad_1

        '''
        self._W1 = self._W1 + self._step_size \
                                    * (reward + discount * np.dot(self._W1, q_idx)
                                       - np.dot(self._W1, q_idx_old)) * q_idx_old
        '''

        # Store current state and action as old state and action.
        self._action = np.copy(action)
        self._s = np.copy(new_state)

        return action

class GreedyTestAgentNeural2(object):
    def __init__(self, W1, W2):
        self._W1 = W1
        self._W2 = W2

    def q_values(self, state):
        h = relu(np.dot(state, self._W1))
        y = sigmoid(np.dot(h, self._W2))
        return y

    def step(self, new_state):
        x = np.reshape(new_state, -1)
        ref_idx = x #+ 1
        empty_spaces = np.where(x == 0)[0].astype('int32')

        q_comparison = np.zeros(len(empty_spaces))
        for a, i in enumerate(empty_spaces):
            q_comparison_idx = np.copy(ref_idx)
            q_comparison_idx[i] = 1 #2 # N.B. 2 is the q-index for a cross
            q_comparison_idx = (q_comparison_idx - np.min(q_comparison_idx)) \
                               / (np.max(q_comparison_idx) - np.min(q_comparison_idx))
            q_comparison[a] = self.q_values(q_comparison_idx)

        q_argmax = np.argwhere(q_comparison == np.amax(q_comparison))
        q_argmax = q_argmax[:, -1]
        if np.shape(q_argmax) == (1,):
            action = empty_spaces[q_argmax[0]]
        else:
            action = np.random.choice(empty_spaces[q_argmax])
        return action