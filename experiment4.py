# 2-LAYER NN FOR APPROXIMATING Q

import numpy as np
import matplotlib.pyplot as plt
from auxiliary import *
from environment import *
from agent_tabular import *

def run_experiment(env, agent, number_of_steps):
    mean_reward = 0.

    initial_action = 0
    action = initial_action
    for i in range(number_of_steps):
        new_state, reward, discount = env.step(action)
        action = agent.step(new_state, reward, discount)
        mean_reward += (reward - mean_reward)/(i + 1.)
    return mean_reward

initial_state = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
initial_action = 0
step_size = 0.1

env = Environment(initial_state)
agent = RandomAgentNeural2(initial_state, initial_action, step_size)

gred = GreedyTestAgentNeural2(agent._W1, agent._W2)
env.step(0)
num_games = 0
num_won = 0
for i in range(int(1e4)):
    gred.step(env._state)
    n, r, g = env.step(gred.step(env._state))
    if g == 0:
        num_games += 1
    if r == 5:
        num_won += 1
print(num_won/num_games)

env = Environment(initial_state)
mean_reward = run_experiment(env, agent, int(1e5))
#q = agent.q_values.reshape(grid._layout.shape + (4,))
#plot_action_values(q)

env = Environment(initial_state)
gred = GreedyTestAgentNeural2(agent._W1, agent._W2)
env.step(0)
num_games = 0
num_won = 0
for i in range(int(1e4)):
    gred.step(env._state)
    n, r, g = env.step(gred.step(env._state))
    if g == 0:
        num_games += 1
    if r == 5:
        num_won += 1
print(num_won/num_games)

# TEST
# env = Environment()
# gred = GreedyTestAgent(agent._q)
# env.step(0)
# gred.step(env._state) # look at what his action is, for curiosity.
# env.step(gred.step(env._state)) # perform the action.

# np.unravel_index(agent._q.argmax(), agent._q.shape)