# Import useful libraries.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import generate_data

# Global variables.
log_period_samples = 40
batch_size = 10

# Import dataset with one-hot encoding of the class labels.
X_train, y_train, X_test, y_test = generate_data.gen()

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.
def get_placeholders():
  x = tf.placeholder(tf.float32, [None, 128,128,3])
  y_ = tf.placeholder(tf.float32, [None, 2])
  return x, y_

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
settings = [(10, 0.1)]

# MODEL 4
# For each of the three hyper-parameter settings:
#   1. The neural network computational graph is defined in the SETUP section.
#   2. Next, this computational graph is trained in the TRAINING section, which
#      updates the values of the variables (trainable parameters).
#   3. Periodically, during training, the EVALUATION section is executed,
#      whereby the computational graph with the current trained values of the
#      parameters is run on an evaluation sets of training and test data.

print('Training Model 4')

# Train Model 4 with the different hyper-parameter settings.
for (num_epochs, learning_rate) in settings:

    # Reset graph, recreate placeholders and dataset.
    tf.reset_default_graph()
    x, y_ = get_placeholders()

    ######################################################################
    # SETUP: Define model, loss, optimiser, training update,             #
    #        and evaluation metric (define computational graph).         #
    ######################################################################

    # Define model architecture:
    #   - two convolutional and max pooling layers, with ReLU activations
    #   - single hidden linear layer with ReLU activation
    #   - single linear layer with softmax activation
    #   - glorot initialisation of all parameters
    initializer = tf.contrib.layers.xavier_initializer()
    W_conv1 = tf.Variable(initializer(shape=[3, 3, 3, 8]))
    b_conv1 = tf.Variable(initializer(shape=[8]))
    W_conv2 = tf.Variable(initializer(shape=[3, 3, 8, 8]))
    b_conv2 = tf.Variable(initializer(shape=[8]))
    W_fc1 = tf.Variable(initializer(shape=[32 * 32 * 8, 32]))
    b_fc1 = tf.Variable(initializer(shape=[32]))
    W_fc2 = tf.Variable(initializer(shape=[32, 2]))
    b_fc2 = tf.Variable(initializer(shape=[2]))

    # Reshape the 1-dimensional input to 28x28 2-dimensional input.
    x_2D = x #tf.reshape(x, [-1, 28, 28, 1])

    # Convolutional layers.
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_2D, W_conv1, strides=[1, 1, 1, 1],
                                      padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                                      padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

    # Fully-connected layers.
    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 8])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    # Define loss function: cross-entropy
    #
    # IMPORTANT NOTE: We will minimise the TOTAL cross-entropy across batch rather
    #                 than MEAN cross-entropy across batch, as instructed to do so
    #                 in the guidelines.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    total_cross_entropy = tf.reduce_sum(cross_entropy)

    # Define optimiser: SGD
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # Define training update: SGD optimiser applied for total cross-entropy loss
    train_step = optimizer.minimize(total_cross_entropy)

    # Define evaluation metric: accuracy (mean across batch)
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #########################################################################
    # TRAINING: Train the model (run computational graph on training data). #
    #########################################################################

    # Set up training session.
    i, ii, train_accuracy, test_accuracy = 0, 0, [], []
    log_period_updates = int(log_period_samples / batch_size)
    with tf.train.MonitoredSession() as sess:
        while i < (num_epochs*800)//batch_size:

            # Track training step number.
            i += 1
            ii += 1

            # Acquire new shuffled batch.

            batch_xs = X_train[batch_size*(ii-1):(ii)*batch_size]
            batch_ys = y_train[batch_size*(ii-1):(ii)*batch_size]

            # Perform training step.
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # Evaluate model, if training step is a designated logging step.
            if i % log_period_updates == 0:
                ####################################################################
                # EVALUATION: Compute and store train & test accuracy              #
                #             (run trained computational graph on evaluation data) #
                ####################################################################

                # Append training accuracy to corresponding list.
                # NOTE: the first 20% of the evaluation training dataset is arbitrarily
                #       chosen for the evaluation; for every evaluation across every
                #       model and setting, the same set is used to ensure consistency.
                train_accuracy.append(sess.run(accuracy, feed_dict=
                {x: X_train,
                 y_: y_train}))

                # Append test accuracy to corrresponding list.
                test_accuracy.append(sess.run(accuracy, feed_dict=
                {x: X_test,
                 y_: y_test}))

            if ii%8 == 0:
                ii = 0


    # Once a setting has been fully trained, append its results to the task list.
    experiments_task4.append(
        ((num_epochs, learning_rate), train_accuracy, test_accuracy))

plot_summary_table([experiments_task4])
plt.show()

plot_learning_curves([experiments_task4])
plt.show()