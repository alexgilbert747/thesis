import numpy as np
import matplotlib.pyplot as plt
from auxiliary import *


class RandomAgent(object):

    def __init__(self, initial_state, initial_action, step_size):
        # start state (2D state)
        self._s = initial_state
        #self._number_of_states = len(np.where(self._s == 0)[0])

        #x = np.reshape(self._initial_state, -1)
        #empty_spaces = np.where(x == 0)[0].astype('int32')
        self._action = initial_action
        self._step_size = step_size

        self._q = np.zeros((3, 3, 3,
                            3, 3, 3,
                            3, 3, 3))


    def step(self, new_state, reward, discount):
        action_old = self._action
        state_old = self._s

        x = np.reshape(new_state, -1)
        ref_idx = x + 1
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
        q_idx[action] = 2


        x_old = np.reshape(state_old, -1)
        ref_idx_old = x_old + 1

        q_idx_old = np.copy(ref_idx_old)
        q_idx_old[action_old] = 2


        self._q[tuple(q_idx_old)] = self._q[tuple(q_idx_old)] + self._step_size \
                                    * (reward + discount * self._q[tuple(q_idx)]
                                       - self._q[tuple(q_idx_old)])



        # Store current state and action as old state and action.
        self._action = np.copy(action)
        self._s = np.copy(new_state)

        return action

class GreedyTestAgent(object):
    def __init__(self, q):
        self._q = q

    def q_values(self):
        pass

    def step(self, new_state):
        x = np.reshape(new_state, -1)
        ref_idx = x + 1
        empty_spaces = np.where(x == 0)[0].astype('int32')

        q_comparison = np.zeros(len(empty_spaces))
        for a, i in enumerate(empty_spaces):
            q_comparison_idx = np.copy(ref_idx)
            q_comparison_idx[i] = 2 # N.B. 2 is the q-index for a cross
            q_comparison[a] = self._q[tuple(q_comparison_idx)]

        q_argmax = np.argwhere(q_comparison == np.amax(q_comparison))
        q_argmax = q_argmax[:, -1]
        if np.shape(q_argmax) == (1,):
            action = empty_spaces[q_argmax[0]]
        else:
            action = np.random.choice(empty_spaces[q_argmax])
        return action