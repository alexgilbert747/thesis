# IMPLEMENT TABULAR NOUGHTS AND CROSSES TOO, SO I CAN MAKE A SIMPLE COMPARISON.

import numpy as np
import matplotlib.pyplot as plt
from auxiliary import *

# Set training hyperparameters.
num_epochs = 1
learning_rate = 0.1

# Set model hyperparameters.

# Initialise parameters.
W1 = np.random.randn(9, 4)
W2 = np.random.randn(4, 1)

def check_win(participant, state):
    win = 0
    for i,j,k in [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]:
        if state[i]+state[j]+state[k] == 3*participant:
            win = 1
    return win

def opponent(state):
    empty_spaces = np.where(state == 0)[0]
    return np.random.choice(empty_spaces)

def player(state, reward, discount):
    #W1 = W1*1
    #W2 = W2*1
    empty_spaces = np.where(state == 0)[0]
    player_action =  np.random.choice(empty_spaces)
    return player_action

def state_update(marker, old_state, old_action):
    new_state = np.copy(old_state)
    new_state[old_action] = marker
    return new_state

def environment(old_state, player_marker, player_action):

    end_flag = 0

    # Non-terminal outputs.
    state = state_update(player_marker, old_state, player_action) #state[player_action] = player_marker
    reward = 0
    discount = 0.9

    # Check if player has won.
    player_win = check_win(player_marker, state)
    if end_flag == 0 and player_win == 1:
        state = np.zeros(9)
        reward = 1
        discount = 0
        end_flag = 1

    # If not and board is not full, opponent plays move.
    if end_flag == 0 and np.prod(state) == 0:
        # Opponent chooses move.
        opponent_action = opponent(state)
        # Update state.
        state[opponent_action] = (-1)*player_marker
        end_flag = 1

    # Check if opponent has won.
    opponent_win = check_win((-1)*player_marker, state)
    if end_flag == 0 and opponent_win == 1:
        state = np.zeros(9)
        reward = -1
        discount = 0
        end_flag = 1

    # Check if board is full and is a draw.
    if end_flag == 0 and player_win == 0 and opponent_win == 0 and np.prod(state) != 0:
        state = np.zeros(9)
        reward = 0
        discount = 0

    return state, reward, discount


num_games = 1
for i in range(num_games):
    old_state = np.zeros(9)
    old_reward = 0
    old_discount = 0
    player_marker = 1
    history_state = []
    history_reward = []
    history_discount = []

    history_state.append(old_state)
    history_reward.append(old_reward)
    history_discount.append(old_discount)

    new_discount = 1
    while new_discount != 0:
        action = player(old_state, old_reward, old_discount)
        state = state_update(player_marker, old_state, action)
        new_state, new_reward, new_discount = environment(np.copy(old_state), np.copy(player_marker), np.copy(action))
        history_state.append(new_state)
        history_reward.append(new_reward)
        history_discount.append(new_discount)
        old_state = new_state
        old_reward = new_reward
        old_discount = new_discount

'''
state = np.zeros(9)
old_state = np.zeros(9)
old_action = 0
player_marker = 1
'''


















