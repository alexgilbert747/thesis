# 1-LAYER NN FOR APPROXIMATING Q
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from auxiliary import *
from environment import *
from agent_neural1 import *

def run_experiment(env, agent, number_of_steps):
    loss_history = []
    cumulative_reward_history = []
    cumulative_reward = 0
    mean_reward = 0
    initial_action = 0
    action = initial_action
    #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
    for i in range(number_of_steps):
        loss_history.append(agent._squared_loss.eval())
        new_state, reward, discount = env.step(action)
        action = agent.step(new_state, reward, discount)
        mean_reward += (reward - mean_reward)/(i + 1.)
        cumulative_reward += reward
        cumulative_reward_history.append(cumulative_reward)
        #loss_history.append(agent._squared_loss.eval())
        '''
        if g == 0:
            loss_history.append(agent._squared_loss.eval())
        '''
    return mean_reward, cumulative_reward_history, loss_history



initial_state = np.array([[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]], dtype='float32')
initial_action = 0
step_size = 0.001
targets = np.load('targets.npy')
targets = np.tanh(targets)#2*((targets - np.min(targets))/(np.max(targets) - np.min(targets))) - 1

with tf.Session() as sess:
    env = Environment(initial_state)
    agent = RandomAgentNeural(initial_state, initial_action, step_size, sess, targets)

    sess.run(tf.global_variables_initializer())
    W1 = agent._W_fc1.eval()
    b1 = agent._b_fc1.eval()
    W2 = agent._W_fc2.eval()
    b2 = agent._b_fc2.eval()
    W3 = agent._W_fc3.eval()
    b3 = agent._b_fc3.eval()
    #print(W1, b1, W2, b2)

    gred = GreedyTestAgentNeural(W1, b1, W2, b2, W3, b3, sess)
    env.step(0)
    num_games = 0
    num_won = 0
    for i in range(int(5e3)):
        #gred.step(env._state)
        n, r, g = env.step(gred.step(env._state))
        if g == 0:
            num_games += 1
        if r == 5:
            num_won += 1
    print(num_won/num_games)

    env = Environment(initial_state)
    mean_reward, cumulative_reward_history, loss_history = run_experiment(env, agent, int(1e4))

# IMPLEMENT TOTAL REGRET CURVE. ALSO CHECK BELOW WHAT THOSE TWO COMMENTS ARE.
#q = agent.q_values.reshape(grid._layout.shape + (4,))
#plot_action_values(q)

    W1 = agent._W_fc1.eval()
    b1 = agent._b_fc1.eval()
    W2 = agent._W_fc2.eval()
    b2 = agent._b_fc2.eval()
    W3 = agent._W_fc3.eval()
    b23 = agent._b_fc3.eval()
    #print(W1, b1, W2, b2)

    env = Environment(initial_state)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #W1 = agent._W_fc1.eval()
    #b1 = agent._b_fc1.eval()
    #print(W1, b1, W2, b2, W3, b3)

    gred = GreedyTestAgentNeural(W1, b1, W2, b2, W3, b3, sess)
    env.step(0)
    num_games = 0
    num_won = 0
    for i in range(int(5e3)):
        #gred.step(env._state)
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