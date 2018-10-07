# Import useful libraries.
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Global variables.
log_period_samples = 20000
batch_size = 100

# Import dataset with one-hot encoding of the class labels.
def get_data():
  return input_data.read_data_sets("MNIST_data/", one_hot=True)

# Plot learning curves of experiments
def plot_learning_curves(experiment_data):
  # Generate figure.
  fig, axes = plt.subplots(3, 4, figsize=(22,12))
  st = fig.suptitle(
      "Learning Curves for all Tasks and Hyper-parameter settings",
      fontsize="x-large")
  # Plot all learning curves.
  for i, results in enumerate(experiment_data):
    for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
      # Plot.
      xs = [x * log_period_samples for x in range(1, len(train_accuracy)+1)]
      axes[j, i].plot(xs, train_accuracy, label='train_accuracy')
      axes[j, i].plot(xs, test_accuracy, label='test_accuracy')
      # Prettify individual plots.
      axes[j, i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
      axes[j, i].set_xlabel('Number of samples processed')
      axes[j, i].set_ylabel('Epochs: {}, Learning rate: {}.  Accuracy'.format(*setting))
      axes[j, i].set_title('Task {}'.format(i + 1))
      axes[j, i].legend()
  # Prettify overall figure.
  plt.tight_layout()
  st.set_y(0.95)
  fig.subplots_adjust(top=0.91)
  plt.show()

# Generate summary table of results.
def plot_summary_table(experiment_data):
  # Fill Data.
  cell_text = []
  rows = []
  columns = ['Setting 1', 'Setting 2', 'Setting 3']
  for i, results in enumerate(experiment_data):
    rows.append('Model {}'.format(i + 1))
    cell_text.append([])
    for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
      cell_text[i].append(test_accuracy[-1])
  # Generate Table.
  fig=plt.figure(frameon=False)
  ax = plt.gca()
  the_table = ax.table(
      cellText=cell_text,
      rowLabels=rows,
      colLabels=columns,
      loc='center')
  the_table.scale(1, 4)
  # Prettify.
  ax.patch.set_facecolor('None')
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)


# CAREFUL: Running this CL resets the experiments_task4 dictionary where results should be stored.
# Store results of runs with different configurations in a dictionary.
# Use a tuple (num_epochs, learning_rate) as keys, and a tuple (training_accuracy, testing_accuracy)
experiments_task4 = []
#settings = [(10, 0.001)]
settings = [(2, 0.001)]

# MODEL 4
# For each of the three hyper-parameter settings:
#   1. The neural network model is defined in the SETUP section.
#   2. Next, this model is trained in the TRAINING section, which
#      updates the values of the variables (trainable parameters).
#   3. Periodically, during training, the EVALUATION section is executed,
#      whereby the model with the current trained values of the
#      parameters is run on evaluation sets of training and test data.

print('Training Model 4')

# Train Model 4 with the different hyper-parameter settings.
for (num_epochs, learning_rate) in settings:

    # Get data (training, training evaluation, test evaluation).
    mnist = get_data()
    eval_mnist = get_data()


    #########################################################################
    # SETUP: Define activations, entire convolutional operation function,   #
    #        layers, combined model, gradient clipping, and regularisation. #
    #########################################################################

    # Define activation functions.
    def softmax(a):
        output = np.exp(a - np.max(a, 1)[:, np.newaxis])  # use x-max(x) to avoid over/underflow
        row_sum = np.sum(output, 1)
        return output / row_sum[:, np.newaxis]


    def relu(a):
        a[a < 0] = 0.
        return a


    # Define entire convolutional layer operation.

    # I found it pretty much impossible, given the time limits, to implement an
    # efficient numpy matrix multiplication scheme for performing fast convolutions, as
    # requested in the guidelines. Therefore, in order to proceed, I have had to use
    # this open-source code provided by botcs on stackoverflow:
    # https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    # His code is contained within the two hash lines below.
    ################################################################################
    def create_x_col(x, filter_size, stride_size):
        # Parameters
        A, B, skip = x, filter_size, stride_size
        batch, M, N, D = A.shape
        col_extent = N - B[1] + 1
        row_extent = M - B[0] + 1

        # Get batch block indices
        batch_idx = np.arange(batch)[:, None, None] * D * M * N

        # Get Starting block indices
        start_idx = np.arange(B[0])[None, :, None] * N + np.arange(B[1])

        # Generate Depth indeces
        didx = M * N * np.arange(D)
        start_idx = (didx[None, :, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[None, :, None] * N + np.arange(col_extent)

        # Get all actual indices & index into input array for final output
        act_idx = (batch_idx +
                   start_idx.ravel()[None, :, None] +
                   offset_idx[:, ::skip[0], ::skip[1]].ravel())

        out = np.take(A, act_idx)
        return out


    ################################################################################

    def create_W_row(W):
        W_row = np.reshape(W, (3 * 3 * np.shape(W)[2], np.shape(W)[3]))
        return W_row


    def matrix_convolution(x_col, W_row):
        return np.dot(np.transpose(x_col, axes=[0, 2, 1]), W_row)


    # Define model architecture:
    #   - single convolutional and max pooling layer
    #   - single convolutional and max pooling layer
    #   - single linear layer with relu activation
    #   - single linear layer with softmax activation

    def make_input_4D(x):
        x = np.reshape(x, (batch_size, 28, 28, 1))
        return x


    def conv_layer1(x):
        x_col = create_x_col(x, (3, 3), (1, 1))
        W_row = create_W_row(W1)
        out1 = matrix_convolution(x_col, W_row) + b1
        return out1


    def reshape_to_4D_1(conv_result):
        return np.reshape(conv_result, (batch_size, 26, 26, 8))


    def max_pooling1(x):
        x = np.reshape(x, (batch_size * 8, 26, 26, 1))
        x_col = create_x_col(x, (2, 2), (2, 2))
        idx = np.argmax(x_col, axis=1)
        x_col_pool = np.amax(x_col, axis=1)
        x_col_pool = x_col_pool[:, np.newaxis, :]
        x_pool = np.reshape(x_col_pool, (batch_size, 13, 13, 8))
        return x_pool, idx


    def conv_layer2(x):
        x_col = create_x_col(x, (3, 3), (1, 1))
        W_row = create_W_row(W2)
        out2 = matrix_convolution(x_col, W_row) + b2
        return out2


    def reshape_to_4D_2(conv_result):
        return np.reshape(conv_result, (batch_size, 11, 11, 8))


    def even_crop_to_allow_pooling(x):
        x = x[:, 0:10, 0:10, :]
        return x


    def max_pooling2(x):
        x = np.reshape(x, (batch_size * 8, 10, 10, 1))
        x_col = create_x_col(x, (2, 2), (2, 2))
        idx = np.argmax(x_col, axis=1)
        x_col_pool = np.amax(x_col, axis=1)
        x_col_pool = x_col_pool[:, np.newaxis, :]
        x_pool = np.reshape(x_col_pool, (batch_size, 5, 5, 8))
        return x_pool, idx


    def flatten(x):
        x_flat = np.reshape(x, (batch_size, 200))
        return x_flat


    def fc_layer1(x):
        out3 = relu(x.dot(W3) + b3)
        return out3


    def fc_layer2(x):
        y_hat = softmax(x.dot(W4) + b4)
        return y_hat

        # - combined model


    def model(x):
        x = make_input_4D(x)
        z1 = conv_layer1(x)
        z1 = reshape_to_4D_1(z1)
        z1, idx = max_pooling1(z1)
        z2 = conv_layer2(z1)
        z2 = reshape_to_4D_2(z2)
        z2 = even_crop_to_allow_pooling(z2)
        z2, idx = max_pooling2(z2)
        z2 = flatten(z2)
        z3 = fc_layer1(z2)
        y_hat = fc_layer2(z3)
        return y_hat


    # Define gradient clipping procedure.
    # NOTE: I believe TensorFlow's optimizer.minimize() function actually applies
    #       automatic gradient clipping for the specified optimizer; therefore,
    #       in order to attempt to mirror the TensorFlow assignment, I define my
    #       own gradient clipping procedure here, which helps numerical stability.
    # NOTE: My chosen procedure is adopted from Pascanu et al. (2013):
    #       https://arxiv.org/pdf/1211.5063.pdf
    #       The threshold I selected is simply from experimenting with different
    #       values; alternatively, one can make the threshold adaptive during
    #       training, but that is trickier to implement.
    def grad_clip(grad):
        threshold = 12
        grad_norm = np.linalg.norm(grad)
        if grad_norm > threshold:
            grad = (threshold / grad_norm) * grad
        return grad


    # Define gradient regularisation procedure.
    # NOTE: I am not sure whether TensorFlow implements this with its in-built
    #       optimisation procedure, but looking at the graphs from the previous
    #       assignment, it seems that it was much more capable of controlling the
    #       growth of the weights as the epochs progressed; as such, I suspected
    #       they use some sort of in-built regularisation, which is why I have
    #       implemented L2 regularisation for my model.
    def grad_reg(grad, W):
        reg_lambda = 0.001
        grad = grad + reg_lambda * W
        return grad


    ##############################
    # TRAINING: Train the model. #
    ##############################

    # Set up training arrays.
    i, train_accuracy, test_accuracy = 0, [], []
    log_period_updates = int(log_period_samples / batch_size)

    # Initialise parameters (Xavier initialisation).
    # NOTE: Input - 28 x 28 x 1
    #       Conv1 - 26 x 26 x 8 ; W1 - 3x3x1x8
    #       Pool  - 13 x 13 x 8
    #       Conv2 - 11 x 11 x 8 ; W2 - 3x3x8x8
    #       crop!   10 x 10 x 8
    #       Pool    5  x 5  x 8
    #       FC1   -
    #       FC2   -
    #
    W1 = np.random.normal(0, np.sqrt(2 / (784 + 5408)), (3, 3, 1, 8))
    b1 = np.zeros((8))
    W2 = np.random.normal(0, np.sqrt(2 / (1352 + 968)), (3, 3, 8, 8))
    b2 = np.zeros((8))

    W3 = np.random.normal(0, np.sqrt(2 / (200 + 32)), (200, 32))
    b3 = np.zeros((1, 32))
    W4 = np.random.normal(0, np.sqrt(2 / (32 + 10)), (32, 10))
    b4 = np.zeros((1, 10))

    # Begin training.
    while mnist.train.epochs_completed < num_epochs:

        # Track training step number.
        i += 1

        # Acquire new shuffled batch.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        x = batch_xs
        y = batch_ys

        # Forward pass:
        x = make_input_4D(x)
        z1 = conv_layer1(x)
        z1 = reshape_to_4D_1(z1)
        z1, idx1 = max_pooling1(z1)
        z2 = conv_layer2(z1)
        z2 = reshape_to_4D_2(z2)
        z2 = even_crop_to_allow_pooling(z2)
        z2, idx2 = max_pooling2(z2)
        z2 = flatten(z2)
        z3 = fc_layer1(z2)
        y_hat = fc_layer2(z3)

        # Backward pass:
        # 1a. Gradients and update for output layer.
        dLdz4 = y_hat - y  # pre-activation (softmax-inputs) gradient

        dLdb4 = np.sum(dLdz4, 0)  # bias gradients (sum loss over batch)
        dLdW4 = np.transpose(z3).dot(dLdz4)  # weight gradients (sum loss over batch)

        dLdb4 = grad_clip(dLdb4)  # clip bias gradients above the threshold
        dLdW4 = grad_clip(dLdW4)  # clip weight gradients above the threshold

        b4 = b4 - learning_rate * dLdb4  # update biases
        W4 = W4 - learning_rate * dLdW4  # update weights

        # 1b. Propagate gradients wrt next lower-level layer's activation.
        dLdz3 = dLdz4.dot(np.transpose(W4))

        # 2a. Gradients and update for third layer.
        dLdz3 = np.multiply(dLdz3, z3 != 0)  # pre-activation (relu-inputs) gradient

        dLdb3 = np.sum(dLdz3, 0)  # bias gradients (sum loss over batch)
        dLdW3 = np.transpose(z2).dot(dLdz3)  # weight gradients (sum loss over batch)

        dLdb3 = grad_clip(dLdb3)  # clip bias gradients above the threshold
        dLdW3 = grad_clip(dLdW3)  # clip weight gradients above the threshold

        b3 = b3 - learning_rate * dLdb3  # update biases
        W3 = W3 - learning_rate * dLdW3  # update weights

        # 2b. Propagate gradients wrt next lower-level layer's activation.
        dLdz2 = dLdz3.dot(np.transpose(W3))

        ############ FAIL STARTS HERE #############

        '''
        # 3a. Gradients and update for second layer.
    
        dLdz2 = np.multiply(dLdz2, z2!=0) # pre-activation (relu-inputs) gradient
    
        dLdb2 = np.sum(dLdz2, 0) # bias gradients (sum loss over batch)
        dLdW2 = np.transpose(z1).dot(dLdz2) # weight gradients (sum loss over batch)
    
        dLdb2 = grad_clip(dLdb2) # clip bias gradients above the threshold
        dLdW2 = grad_clip(dLdW2) # clip weight gradients above the threshold
    
        b2 = b2 - learning_rate * dLdb2 # update biases
        W2 = W2 - learning_rate * dLdW2 # update weights
    
        # 3b. Propagate gradients wrt next lower-level layer's activation.
        dLdz1 = dLdz2.dot(np.transpose(W2)) 
    
        # 4a. Gradients and update for first layer.
        dLdz1 = np.multiply(dLdz1, z1!=0) # pre-activation (relu-inputs) gradient
    
        dLdb1 = np.sum(dLdz1, 0) # bias gradients (sum loss over batch)
        dLdW1 = np.transpose(x).dot(dLdz1) # weight gradients (sum loss over batch)
    
        dLdb1 = grad_clip(dLdb1) # clip bias gradients above the threshold
        dLdW1 = grad_clip(dLdW1) # clip weight gradients above the threshold
    
        b1 = b1 - learning_rate * dLdb1 # update biases
        W1 = W1 - learning_rate * dLdW1 # update weights
    
    
        # 4b. End of backward pass; propagation finished.
        #     No further code required.  
    
        '''

        ############## END OF FAIL #############

        # Evaluate model, if training step is a designated logging step.
        if i % log_period_updates == 0:
            #######################################################
            # EVALUATION: Compute and store train & test accuracy #
            #######################################################

            # Append training accuracy to corresponding list.
            # NOTE: the first 20% of the evaluation training dataset is arbitrarily
            #       chosen for the evaluation; for every evaluation across every
            #       model and setting, the same set is used to ensure consistency.
            batch_size = 11000
            y_hat = model(eval_mnist.train.images[0:11000, :])
            y = eval_mnist.train.labels[0:11000, :]
            num_correct = np.sum(np.argmax(y, 1) == np.argmax(y_hat, 1))
            accuracy = num_correct / np.shape(y)[0]
            train_accuracy.append(accuracy)

            # Append test accuracy to corrresponding list.
            batch_size = 10000
            y_hat = model(eval_mnist.test.images)
            y = eval_mnist.test.labels
            num_correct = np.sum(np.argmax(y, 1) == np.argmax(y_hat, 1))
            accuracy = num_correct / np.shape(y)[0]
            test_accuracy.append(accuracy)

            batch_size = 100
    # Once a setting has been fully trained, append its results to the task list.
    experiments_task4.append(
        ((num_epochs, learning_rate), train_accuracy, test_accuracy))




plot_learning_curves([experiments_task4])