import numpy as np

class Environment(object):

    def __init__(self, initial_state):
        # start state (2D state)
        self._state = initial_state
        #self._number_of_states = len(np.where(self._state == 0)[0])

        # 3-in-a-row win conditions (for flattened 1D state)
        self._win_combos = \
            [[0, 1, 2], [3, 4, 5], [6, 7, 8],
             [0, 3, 6], [1, 4, 7], [2, 5, 8],
             [0, 4, 8], [2, 4, 6]]

    def opponent_random(self, x):
        empty_spaces = np.where(x == 0)[0].astype('int32')
        return np.random.choice(empty_spaces)

    def opponent_greedy(self, x, q_current):
        ref_idx = x + 1
        empty_spaces = np.where(x == 0)[0].astype('int32')

        q_comparison = np.zeros(len(empty_spaces))
        for a, i in enumerate(empty_spaces):
            q_comparison_idx = np.copy(ref_idx)
            q_comparison_idx[i] = 2  # N.B. 2 is the q-index for a cross
            q_comparison[a] = q_current[tuple(q_comparison_idx)]

        q_argmax = np.argwhere(q_comparison == np.amax(q_comparison))
        q_argmax = q_argmax[:, -1]
        if np.shape(q_argmax) == (1,):
            action = empty_spaces[q_argmax[0]]
        else:
            action = np.random.choice(empty_spaces[q_argmax])
        return action

    def check_win(self, marker, x):
        win = False
        for i, j, k in self._win_combos:
            if x[i] + x[j] + x[k] == 3*marker:
                win = True
        return win

    def check_draw(self, x):
        return np.prod(x) != 0

    def step(self, action):
        x = np.reshape(self._state, -1)

        # Place player's marker on board.
        x[action] = 1

        # Check win condition for player.
        win = self.check_win(1, x)
        if win == True:
            x = np.zeros(9, dtype='int32')
            reward = 5
            discount = 0

        # Check draw condition.
        else:
            draw = self.check_draw(x)
            if draw == True:
                x = np.zeros(9, dtype='int32')
                reward = 0
                discount = 0

            # Opponent's move: place opponent's marker on board.
            else:
                opponent_action = self.opponent_random(x) #self.opponent_greedy(x, q_current)
                # Update state.
                x[opponent_action] = -1
                reward = 0
                discount = 0.9

                # Check win condition for opponent.
                win = self.check_win(-1, x)
                if win == True:
                    x = np.zeros(9, dtype='int32')
                    reward = -5
                    discount = 0

                # Check draw condition.
                else:
                    draw = self.check_draw(x)
                    if draw == True:
                        x = np.zeros(9, dtype='int32')
                        reward = 0
                        discount = 0
                    else:
                        reward = 0
                        discount = 0.9

        self._state = np.reshape(x, (3, 3))

        return self._state, reward, discount