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


# CAREFUL: Running this CL resets the experiments_task3 dictionary where results should be stored.
# Store results of runs with different configurations in a dictionary.
# Use a tuple (num_epochs, learning_rate) as keys, and a tuple (training_accuracy, testing_accuracy)
experiments_task3 = []
settings = [(40, 0.003)]

# MODEL 3
# For each of the three hyper-parameter settings:
#   1. The neural network model is defined in the SETUP section.
#   2. Next, this model is trained in the TRAINING section, which
#      updates the values of the variables (trainable parameters).
#   3. Periodically, during training, the EVALUATION section is executed,
#      whereby the model with the current trained values of the
#      parameters is run on evaluation sets of training and test data.

print('Training Model 3')

# Train Model 3 with the different hyper-parameter settings.
for (num_epochs, learning_rate) in settings:

    # Get data (training, training evaluation, test evaluation).
    mnist = get_data()
    eval_mnist = get_data()


    #########################################################################
    # SETUP: Define activations, layers, combined model, gradient clipping, #
    #         and regularisation.                                           #
    #########################################################################

    # Define activation functions.
    def softmax(a):
        output = np.exp(a - np.max(a, 1)[:, np.newaxis])  # use x-max(x) to avoid over/underflow
        row_sum = np.sum(output, 1)
        return output / row_sum[:, np.newaxis]


    def relu(a):
        a[a < 0] = 0.
        return a


    # Define model architecture:
    #   - single linear layer with relu activation
    #   - single linear layer with relu activation
    #   - single linear layer with softmax activation

    def layer1(x):
        z1 = relu(x.dot(W1) + b1)
        return z1


    def layer2(z1):
        z2 = relu(z1.dot(W2) + b2)
        return z2


    def layer3(z2):
        y_hat = softmax(z2.dot(W3) + b3)
        return y_hat

        # - combined model


    def model(x):
        y_hat = layer3(layer2(layer1(x)))
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
        threshold = 8
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
    W1 = np.random.normal(0, np.sqrt(2 / (784 + 32)), (784, 32))
    b1 = np.zeros((1, 32))
    W2 = np.random.normal(0, np.sqrt(2 / (32 + 32)), (32, 32))
    b2 = np.zeros((1, 32))
    W3 = np.random.normal(0, np.sqrt(2 / (32 + 10)), (32, 10))
    b3 = np.zeros((1, 10))

    # Begin training.
    while mnist.train.epochs_completed < num_epochs:

        # Track training step number.
        i += 1

        # Acquire new shuffled batch.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        x = batch_xs
        y = batch_ys

        # Forward pass:
        z1 = layer1(x)
        z2 = layer2(z1)
        y_hat = layer3(z2)

        # Backward pass:
        # 1a. Gradients and update for output layer.
        dLdz3 = y_hat - y  # pre-activation (softmax-inputs) gradient

        dLdb3 = np.sum(dLdz3, 0)  # bias gradients (sum loss over batch)
        dLdW3 = np.transpose(z2).dot(dLdz3)  # weight gradients (sum loss over batch)

        dLdW3 = grad_reg(dLdW3, W3)  # L2 regularise weight gradients

        dLdb3 = grad_clip(dLdb3)  # clip bias gradients above the threshold
        dLdW3 = grad_clip(dLdW3)  # clip weight gradients above the threshold

        b3 = b3 - learning_rate * dLdb3  # update biases
        W3 = W3 - learning_rate * dLdW3  # update weights

        # 1b. Propagate gradients wrt next lower-level layer's activation.
        dLdz2 = dLdz3.dot(np.transpose(W3))

        # 2a. Gradients and update for second layer.
        dLdz2 = np.multiply(dLdz2, z2 != 0)  # pre-activation (relu-inputs) gradient

        dLdb2 = np.sum(dLdz2, 0)  # bias gradients (sum loss over batch)
        dLdW2 = np.transpose(z1).dot(dLdz2)  # weight gradients (sum loss over batch)

        dLdW2 = grad_reg(dLdW2, W2)  # L2 regularise weight gradients

        dLdb2 = grad_clip(dLdb2)  # clip bias gradients above the threshold
        dLdW2 = grad_clip(dLdW2)  # clip weight gradients above the threshold

        b2 = b2 - learning_rate * dLdb2  # update biases
        W2 = W2 - learning_rate * dLdW2  # update weights

        # 2b. Propagate gradients wrt next lower-level layer's activation.
        dLdz1 = dLdz2.dot(np.transpose(W2))

        # 3a. Gradients and update for first layer.
        dLdz1 = np.multiply(dLdz1, z1 != 0)  # pre-activation (relu-inputs) gradient

        dLdb1 = np.sum(dLdz1, 0)  # bias gradients (sum loss over batch)
        dLdW1 = np.transpose(x).dot(dLdz1)  # weight gradients (sum loss over batch)

        dLdW1 = grad_reg(dLdW1, W1)  # L2 regularise weight gradients

        dLdb1 = grad_clip(dLdb1)  # clip bias gradients above the threshold
        dLdW1 = grad_clip(dLdW1)  # clip weight gradients above the threshold

        b1 = b1 - learning_rate * dLdb1  # update biases
        W1 = W1 - learning_rate * dLdW1  # update weights

        # 3b. End of backward pass; propagation finished.
        #     No further code required.

        # Evaluate model, if training step is a designated logging step.
        if i % log_period_updates == 0:
            #######################################################
            # EVALUATION: Compute and store train & test accuracy #
            #######################################################

            # Append training accuracy to corresponding list.
            # NOTE: the first 20% of the evaluation training dataset is arbitrarily
            #       chosen for the evaluation; for every evaluation across every
            #       model and setting, the same set is used to ensure consistency.
            y_hat = model(eval_mnist.train.images[0:11000, :])
            y = eval_mnist.train.labels[0:11000, :]
            num_correct = np.sum(np.argmax(y, 1) == np.argmax(y_hat, 1))
            accuracy = num_correct / np.shape(y)[0]
            train_accuracy.append(accuracy)

            # Append test accuracy to corrresponding list.
            y_hat = model(eval_mnist.test.images)
            y = eval_mnist.test.labels
            num_correct = np.sum(np.argmax(y, 1) == np.argmax(y_hat, 1))
            accuracy = num_correct / np.shape(y)[0]
            test_accuracy.append(accuracy)

    # Once a setting has been fully trained, append its results to the task list.
    experiments_task3.append(
        ((num_epochs, learning_rate), train_accuracy, test_accuracy))

plot_learning_curves([experiments_task3])
plt.show()
plot_summary_table([experiments_task3])
plt.show()