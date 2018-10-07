import numpy as np
from scipy import signal
import mnist
import matplotlib.pyplot as plt
import generate_data

np.random.seed(5)
X_train, y_train, X_test, y_test = generate_data.gen()


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
    for i in range(0,np.shape(x)[0],2):
        for j in range(0,np.shape(x)[1],2):
            avg = (x[i,j] + x[i,j+1] + x[i+1,j] + x[i+1,j+1])/4
            x[i,j] = avg
            x[i,j+1] = avg
            x[i+1,j] = avg
            x[i+1,j+1] = avg
    return x

def max_pool(x):
    x_new = np.zeros((int(np.shape(x)[0]/2), int(np.shape(x)[1]/2), np.shape(x)[2]))
    for b in range(np.shape(x)[2]):
        for i in range(0, np.shape(x_new)[0]):
            for j in range(0, np.shape(x_new)[1]):
                x_new[i,j,b] = np.max([x[2*i,2*j],x[2*i,2*j+1],x[2*i+1,2*j],x[2*i+1,2*j+1]])
    return x_new

from skimage.util.shape import view_as_windows
def strided4D(arr,arr2,s):
    return view_as_windows(arr, arr2.shape, step=s)
def stride_conv_strided(arr,arr2,s):
    arr4D = strided4D(arr,arr2,s=s)
    return np.tensordot(arr4D, arr2, axes=((2,3),(0,1)))

# Set hyperparameters.
num_epoch = 1
learning_rate = 0.1
critical_period_last = 11

# Set weights.
num_filters = 3
num_channels = 3
filter_size = 3
stride = 3
res = np.shape(X_train)[1]
q = (res - filter_size)//stride + 1 #res - (filter_size-1) # compute output dimension (no stride/padding)
w1 = np.random.randn(num_filters, num_channels, filter_size, filter_size)
w2 = np.random.randn(num_filters*21*21, 2) #num_filters*q*q, 2)
#w3 = np.random.randn(1024, 10)

# BIASESSSSSSSSSSSS

X_train_4D = np.reshape(X_train, (len(X_train), res, res, num_channels))
X_test_4D = np.reshape(X_test, (len(X_test), res, res, num_channels))

# Define loss computation.
def compute_loss(X, y):
    num_correct = 0
    for i in range(len(X)):
        # Forward pass:

        # First layer (convolutional).
        layer_1 = np.zeros((q, q, num_filters))
        for j in range(num_filters):
            channel_sum = np.zeros((q, q))
            for k in range(num_channels):
                channel_sum = channel_sum + stride_conv_strided(X[i, :, :, k],w1[j, k, :, :],stride)
                #signal.convolve2d(X[i, :, :, k], w1[j, k, :, :], 'valid')
            layer_1[:,:,j] = channel_sum
        # First layer activation.
        layer_1_pool = max_pool(layer_1)
        layer_1_act = tanh(layer_1_pool)

        # Reshape.
        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)

        # Second layer (fully-connected).
        layer_2 = layer_1_act_vec.dot(w2)
        # Second layer activation.
        layer_2_act = softmax(layer_2)

        # Update number of correctly classified examples.
        correct = np.argmax(y[i]) == np.argmax(layer_2_act)
        num_correct = num_correct + correct
    # Compute accuracy.
    accuracy = num_correct/len(X)
    return accuracy

# Compute training loss before training.
train_accuracy_before = compute_loss(X_train_4D, y_train)
print(train_accuracy_before)

# Compute test loss before training.
test_accuracy_before = compute_loss(X_test_4D, y_test)
print(test_accuracy_before)

# ----- TRAINING -------
for iter in range(num_epoch):
    for i in range(len(X_train_4D)):
            # Forward pass:

            # First layer (convolutional).
            layer_1 = np.zeros((q, q, num_filters))
            for j in range(num_filters):
                channel_sum = np.zeros((q, q))
                for k in range(num_channels):
                    channel_sum = channel_sum + stride_conv_strided(X_train_4D[i, :, :, k],w1[j, k, :, :],stride)
                    #signal.convolve2d(X_train_4D[i, :, :, k], w1[j, k, :, :], 'valid')
                layer_1[:, :, j] = channel_sum
            # First layer activation.
            layer_1_pool = max_pool(layer_1)
            layer_1_act = tanh(layer_1_pool)

            # Reshape.
            layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)

            # Second layer (fully-connected).
            layer_2 = layer_1_act_vec.dot(w2)
            # Second layer activation.
            layer_2_act = softmax(layer_2)

            #cost = np.square(layer_2_act - Y[i]).sum() * 0.5
            # print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost,end="\r")
            # INTERMEDIATE LOSS

            # Backward pass:

            # Second layer gradients.
            grad_2_part_1 = 1
            grad_2_part_2 = layer_2_act - y_train[i]
            grad_2_part_3 = layer_1_act_vec
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)
            w2 = w2 - grad_2 * learning_rate

            # First layer gradients.
            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
            grad_1_part_2 = d_tanh(layer_1_pool)#d_tanh(layer_1)
            grad_1_part_3 = X_train_4D[i,:,:,:]

            grad_1_part_1_reshape = np.reshape(grad_1_part_1, (21, 21, num_filters))
            grad_1 = np.zeros((num_filters, num_channels, filter_size, filter_size))
            for j in range(num_filters):
                grad_1_temp_1 = grad_1_part_1_reshape[:, :, j] * grad_1_part_2[:, :, j]
                for k in range(num_channels):
                    grad_1[j, k, :, :] = np.rot90(
                        stride_conv_strided(grad_1_part_3[:, :, k], np.rot90(grad_1_temp_1, 2), q), 2)

                    #np.rot90(
                    #    signal.convolve2d(grad_1_part_3[:, :, k], np.rot90(grad_1_temp_1, 2), 'valid'), 2)

                    '''
                    if iter > critical_period_last:
                        w1[j,:,:] = np.clip(w1[j,:,:]-grad_1*learning_rate, w1a[j,:,:], w1b[j,:,:])
                    else:
                        w1[j, :, :] = w1[j, :, :] - grad_1 * learning_rate
                    '''
            w1 = w1 - grad_1 * learning_rate
    '''
    if iter == critical_period_last:
        w1_halfspread = np.abs(0.05*w1)
        w1a = w1 - w1_halfspread
        w1b = w1 + w1_halfspread
    '''

# Compute training loss after training.
train_accuracy_after = compute_loss(X_train_4D, y_train)
print(train_accuracy_after)

# Compute test loss after training.
test_accuracy_after = compute_loss(X_test_4D, y_test)
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
