import numpy as np
from scipy import signal
import mnist
import matplotlib.pyplot as plt
import generate_data

np.random.seed(5)
X_train, y_train, X_test, y_test = generate_data.gen()


'''
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def log(x):
    return 1 / (1 + np.exp(-1 * x))

def d_log(x):
    return log(x) * (1 - log(x))

def relu(a):
    a[a < 0] = 0.
    return a

def d_relu(a):
    a[a < 0] = 0.
    a[a > 0] = 1.
    return a

def softmax(a):
    output = np.exp(a - np.max(a, 1)[:, np.newaxis])  # use x-max(x) to avoid over/underflow
    row_sum = np.sum(output, 1)
    return output / row_sum[:, np.newaxis]

def coarsen(x):
    for i in range(0,28,2):
        for j in range(0,28,2):
            avg = (x[i,j] + x[i,j+1] + x[i+1,j] + x[i+1,j+1])/4
            x[i,j] = avg
            x[i,j+1] = avg
            x[i+1,j] = avg
            x[i+1,j+1] = avg
    return x

#mnist.init()
X_train, y_train, X_test, y_test = mnist.load()

# normalise input.
X_train = (X_train - np.min(X_train))/(np.max(X_train) - np.min(X_train))
X_test = (X_test - np.min(X_train))/(np.max(X_train) - np.min(X_train))

# convert labels to one-hot.
one_hot = np.zeros((np.shape(y_train)[0], 10), dtype=int)
for i in range(len(one_hot)):
    one_hot[i][y_train[i]] = 1
y_train = one_hot

one_hot = np.zeros((np.shape(y_test)[0], 10), dtype=int)
for i in range(len(one_hot)):
    one_hot[i][y_test[i]] = 1
y_test = one_hot

#np.random.seed(0)

# 1. Declare hyper Parameters
num_epoch = 10
critical_period_last = 1
learning_rate = 0.1
batch_size = 100

# 0. Declare Weights
filter_size = 1
w1 = np.random.randn(filter_size, 3, 3)
w2 = np.random.randn(filter_size*26*26, 10)
#w3 = np.random.randn(1024, 10)

# BIASESSSSSSSSSSSS

#X_train_3D = np.reshape(X_train, (np.shape(X_train)[0], 28, 28))
#X_test_3D = np.reshape(X_test, (np.shape(X_test)[0], 28, 28))

X_train_4D = np.reshape(X_train, (np.shape(X_train)[0], 28, 28, 1))
X_test_4D = np.reshape(X_test, (np.shape(X_test)[0], 28, 28, 1))

# ---- Cost before training ------
num_correct = 0
for i in range(len(X_train)//batch_size):
    for j in range(batch_size):
        layer_1 = np.zeros((26,26,np.shape(w1)[0]))
        for k in range(np.shape(w1)[0]):
            layer_1[:,:,k] = signal.convolve2d(X_train_4D[batch_size*i+j, :, :, 0], w1[k,:,:], 'valid')

        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = softmax(layer_2)  # layer_2_act = log(layer_2)
        #layer_3 = layer_2_act.dot(w3)
        #layer_3_act = log(layer_3)
        correct = np.argmax(y_train[batch_size*i+j]) == np.argmax(layer_2_act)
        num_correct = num_correct + correct
train_accuracy_before = num_correct/len(X_train)
print(train_accuracy_before)

num_correct = 0
for i in range(len(X_test)//batch_size):
    for j in range(batch_size):
        layer_1 = np.zeros((26,26,np.shape(w1)[0]))
        for k in range(np.shape(w1)[0]):
            layer_1[:,:,k] = signal.convolve2d(X_test_4D[batch_size*i+j,:,:,0], w1[k,:,:], 'valid')

        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = softmax(layer_2)
        #layer_3 = layer_2_act.dot(w3)
        #layer_3_act = log(layer_3)
        correct = np.argmax(y_test[batch_size*i+j]) == np.argmax(layer_2_act)
        num_correct = num_correct + correct
test_accuracy_before = num_correct/len(X_test)
print(test_accuracy_before)

# ----- TRAINING -------
for iter in range(num_epoch):
    for i in range(len(X_test) // batch_size): # CHANGE BACK TO X_train
        for j in range(batch_size):
            if iter < critical_period_last+1:
                x = coarsen(X_train_4D[batch_size * i + j, :, :, 0])
            else:
                x = X_train_4D[batch_size * i + j, :, :, 0]
            layer_1 = np.zeros((26, 26, np.shape(w1)[0]))
            for k in range(np.shape(w1)[0]):
                layer_1[:,:,k] = signal.convolve2d(x, w1[k, :, :], 'valid')

            layer_1_act = tanh(layer_1)

            layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
            layer_2 = layer_1_act_vec.dot(w2)
            layer_2_act = softmax(layer_2)
            #layer_3 = layer_2_act.dot(w3)
            #layer_3_act = log(layer_3)

            #cost = np.square(layer_2_act - Y[i]).sum() * 0.5
            # print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost,end="\r")
            #grad_3_part_1 = layer_3_act - y_train[batch_size*i+j]
            #grad_3_part_2 = d_log(layer_3)
            #grad_3_part_3 = layer_2_act
            #grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)
            #w3 = w3 - grad_3 * learning_rate

            #grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3.T)
            #grad_2_part_2 = d_log(layer_2)
            #grad_2_part_3 = layer_1_act_vec
            #grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
            #w2 = w2 - grad_2 * learning_rate

            grad_2_part_1 = 1 #layer_2_act - y_train[batch_size*i+j]
            grad_2_part_2 = layer_2_act - y_train[batch_size*i+j]#d_log(layer_2)
            grad_2_part_3 = layer_1_act_vec
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
            w2 = w2 - grad_2 * learning_rate


            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
            grad_1_part_2 = d_tanh(layer_1)
            grad_1_part_3 = X_train_4D[batch_size*i+j,:,:,0]

            grad_1_part_1_reshape = np.reshape(grad_1_part_1, (26, 26, np.shape(w1)[0]))
            for k in range(np.shape(w1)[0]):
                grad_1_temp_1 = grad_1_part_1_reshape[:,:,k] * grad_1_part_2[:,:,k]
                grad_1 = np.rot90(
                    signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2), 'valid'),
                    2)
                if iter > critical_period_last:
                    w1[k,:,:] = np.clip(w1[k,:,:]-grad_1*learning_rate, w1a[k,:,:], w1b[k,:,:])
                else:
                    w1[k, :, :] = w1[k, :, :] - grad_1 * learning_rate
    if iter == critical_period_last:
        w1_halfspread = np.abs(0.05*w1)
        w1a = w1 - w1_halfspread
        w1b = w1 + w1_halfspread

num_correct = 0
for i in range(len(X_train)//batch_size):
    for j in range(batch_size):
        layer_1 = np.zeros((26,26,np.shape(w1)[0]))
        for k in range(np.shape(w1)[0]):
            layer_1[:,:,k] = signal.convolve2d(X_train_4D[batch_size*i+j, :, :, 0], w1[k,:,:], 'valid')

        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = softmax(layer_2)
        #layer_3 = layer_2_act.dot(w3)
        #layer_3_act = log(layer_3)
        correct = np.argmax(y_train[batch_size*i+j]) == np.argmax(layer_2_act)
        num_correct = num_correct + correct
train_accuracy_after = num_correct/len(X_train)
print(train_accuracy_after)

num_correct = 0
for i in range(len(X_test)//batch_size):
    for j in range(batch_size):
        layer_1 = np.zeros((26,26,np.shape(w1)[0]))
        for k in range(np.shape(w1)[0]):
            layer_1[:,:,k] = signal.convolve2d(X_test_4D[batch_size*i+j, :, :, 0], w1[k,:,:], 'valid')

        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = softmax(layer_2)
        #layer_3 = layer_2_act.dot(w3)
        #layer_3_act = log(layer_3)
        correct = np.argmax(y_test[batch_size*i+j]) == np.argmax(layer_2_act)
        num_correct = num_correct + correct
test_accuracy_after = num_correct/len(X_test)
print(test_accuracy_after)

"""
# ---- Cost after training ------
for i in range(len(X)):
    layer_1 = signal.convolve2d(X[i], w1, 'valid')
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)
    cost = np.square(layer_2_act - Y[i]).sum() * 0.5
    cost_after_train = cost_after_train + cost
    #final_out = np.append(final_out, layer_2_act)

# ----- Print Results ---
print("\nW1 :", w1, "\n\nw2 :", w2)
print("----------------")
print("Cost before Training: ", cost_before_train)
print("Cost after Training: ", cost_after_train)
print("----------------")
#print("Start Out put : ", start_out)
#print("Final Out put : ", final_out)
#print("Ground Truth  : ", Y.T)
"""
# -- end code --
'''