# IMAGE RECOGNITION - CRITICAL PERIOD NETWORK

# Import useful libraries.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from auxiliary import *
import generate_data
#import pets_data
import pets_data2
import analysis
import visualise

np.random.seed(33)
tf.set_random_seed(33)

num_classA = 694
num_classB = 694
num_examples = num_classA + num_classB
num_class2A = 692
num_class2B = 704
num_examples2 = num_class2A + num_class2B

print('Importing data, normalising, shuffling, and creating train/test splits...')
# Import data.
X_data, y_data, X_data2, y_data2 = pets_data2.gen()

# Normalise data.
X_data = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data))
X_data2 = (X_data2 - np.min(X_data2))/(np.max(X_data2) - np.min(X_data2))

# Global variables.
num_reps = 20
num_epochs = 40 # 50
learning_rate = 0.01 # 0.01 #0.001
batch_size = 40 # 190
log_period_samples = 960 # 760
log_period_updates = int(log_period_samples / batch_size)
constraint_half_range1 = 0.05
constraint_half_range2 = 0.05 # 0.05
constraint_half_range3 = 0.05 # 0.05
constraint_half_range4 = 0.05 # 0.05

res = 64
input_channels = 3

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.
def get_placeholders():
  x = tf.placeholder(tf.float32, [None, res, res, input_channels])
  y = tf.placeholder(tf.float32, [None, 2])
  return x, y

experiment = []

train_accuracies1 = []
test_accuracies1 = []
train_accuracies2 = []
test_accuracies2 = []

for rep in range(num_reps):
    print('EXPERIMENT REPETITION ' + str(rep+1))

    # Shuffle data (to mix up twists and flips within each class).
    random_orderingA = np.arange(0, num_classA)
    random_orderingB = np.arange(num_classA, num_examples)
    np.random.shuffle(random_orderingA)
    np.random.shuffle(random_orderingB)
    X_data[0:num_classA] = X_data[random_orderingA]
    y_data[0:num_classA] = y_data[random_orderingA]
    X_data[num_classA:num_examples] = X_data[random_orderingB]
    y_data[num_classA:num_examples] = y_data[random_orderingB]

    random_ordering2A = np.arange(0, num_class2A)
    random_ordering2B = np.arange(num_class2A, num_examples2)
    np.random.shuffle(random_ordering2A)
    np.random.shuffle(random_ordering2B)
    X_data2[0:num_class2A] = X_data2[random_ordering2A]
    y_data2[0:num_class2A] = y_data2[random_ordering2A]
    X_data2[num_class2A:num_examples2] = X_data2[random_ordering2B]
    y_data2[num_class2A:num_examples2] = y_data2[random_ordering2B]

    # Create splits.
    idx_train = list(range(0, 486)) + list(range(num_classA, 1180))
    idx_test = list(range(486, num_classA)) + list(range(1180, num_classA + num_classB))
    X_train = X_data[idx_train]
    y_train = y_data[idx_train]
    X_test = X_data[idx_test]
    y_test = y_data[idx_test]

    idx_train2 = list(range(0, 489)) + list(range(num_class2A, 1181))
    idx_test2 = list(range(489, num_class2A)) + list(range(1181, num_class2A + num_class2B))
    X_train2 = X_data2[idx_train2]
    y_train2 = y_data2[idx_train2]
    X_test2 = X_data2[idx_test2]
    y_test2 = y_data2[idx_test2]

    # Shuffle train data (to mix up classes).
    random_ordering_train = np.arange(len(X_train))
    np.random.shuffle(random_ordering_train)
    X_train = X_train[random_ordering_train]
    y_train = y_train[random_ordering_train]

    random_ordering_train2 = np.arange(len(X_train2))
    np.random.shuffle(random_ordering_train2)
    X_train2 = X_train2[random_ordering_train2]
    y_train2 = y_train2[random_ordering_train2]

    # VANILLA MODEL
    # For each of the three hyper-parameter settings:
    #   1. The neural network computational graph is defined in the SETUP section.
    #   2. Next, this computational graph is trained in the TRAINING section, which
    #      updates the values of the variables (trainable parameters).
    #   3. Periodically, during training, the EVALUATION section is executed,
    #      whereby the computational graph with the current trained values of the
    #      parameters is run on an evaluation sets of training and test data.

    # Reset graph, recreate placeholders and dataset.
    tf.reset_default_graph()
    x, y = get_placeholders()

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
    W_conv1 = tf.Variable(initializer(shape=[3, 3, input_channels, 4]))
    b_conv1 = tf.Variable(initializer(shape=[4]))
    W_conv2 = tf.Variable(initializer(shape=[5, 5, 4, 8]))
    b_conv2 = tf.Variable(initializer(shape=[8]))
    # W_conv3 = tf.Variable(initializer(shape=[7, 7, 8, 16]))
    # b_conv3 = tf.Variable(initializer(shape=[16]))
    W_fc1 = tf.Variable(initializer(shape=[res // 4 * res // 4 * 8, 512]))
    b_fc1 = tf.Variable(initializer(shape=[512]))
    W_fc2 = tf.Variable(initializer(shape=[512, 2]))
    b_fc2 = tf.Variable(initializer(shape=[2]))
    # W_fc3 = tf.Variable(initializer(shape=[128, 2]))
    # b_fc3 = tf.Variable(initializer(shape=[2]))

    prob = tf.placeholder_with_default(1.0, shape=())

    # Convolutional layers.
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1],
                                      padding='SAME') + b_conv1)
    h_conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
    h_conv1 = tf.nn.dropout(h_conv1, keep_prob=prob)

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1],
                                      padding='SAME') + b_conv2)
    h_conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob=prob)

    # Fully-connected layers.
    h_conv2 = tf.reshape(h_conv2, [-1, res // 4 * res // 4 * 8])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv2, W_fc1) + b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob=prob)
    y_hat = tf.matmul(h_fc1, W_fc2) + b_fc2

    # Define loss function: cross-entropy
    #
    # IMPORTANT NOTE: We will minimise the TOTAL cross-entropy across batch rather
    #                 than MEAN cross-entropy across batch, as instructed to do so
    #                 in the guidelines.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
    total_cross_entropy = tf.reduce_mean(cross_entropy)

    # Define optimiser: SGD
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # Define training update: SGD optimiser applied for total cross-entropy loss
    train_step = optimizer.minimize(total_cross_entropy)

    # Define evaluation metric: accuracy (mean across batch)
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # Operation to save weights.
    saver = tf.train.Saver()

    print('--- Training Critical-Period Model, Task 1 ---')

    #########################################################################
    # TRAINING: Train the model (run computational graph on training data). #
    #########################################################################

    # Set up training session.
    i, n = 0, 0
    train_accuracy1, test_accuracy1 = [], []
    train_accuracy2, test_accuracy2 = [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Initial evaluation.
        train_accuracy1.append(sess.run(accuracy, feed_dict=
        {x: X_train,
         y: y_train}))

        test_accuracy1.append(sess.run(accuracy, feed_dict=
        {x: X_test,
         y: y_test}))

        while i * batch_size < num_epochs * len(X_train):

            # Track training step number.
            i += 1
            n += 1

            # Acquire new shuffled batch.
            batch_xs = X_train[(n - 1) * batch_size:n * batch_size, :, :, :]
            batch_ys = y_train[(n - 1) * batch_size:n * batch_size, :]

            # Perform training step.
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, prob: 0.8})

            # Evaluate model, if training step is a designated logging step.
            if i * batch_size % log_period_samples == 0:
                ####################################################################
                # EVALUATION: Compute and store train & test accuracy              #
                #             (run trained computational graph on evaluation data) #
                ####################################################################

                # Append training accuracy to corresponding list.
                # NOTE: the first 20% of the evaluation training dataset is arbitrarily
                #       chosen for the evaluation; for every evaluation across every
                #       model and setting, the same set is used to ensure consistency.
                train_accuracy1.append(sess.run(accuracy, feed_dict=
                {x: X_train,
                 y: y_train}))

                # Append test accuracy to corrresponding list.
                test_accuracy1.append(sess.run(accuracy, feed_dict=
                {x: X_test,
                 y: y_test}))

            if n % (len(X_train) // batch_size) == 0:
                n = 0
                random_ordering_train = np.arange(len(X_train))
                np.random.shuffle(random_ordering_train)
                X_train = X_train[random_ordering_train]
                y_train = y_train[random_ordering_train]

        W_conv1_s = W_conv1.eval()
        b_conv1_s = b_conv1.eval()
        W_conv2_s = W_conv2.eval()
        b_conv2_s = b_conv2.eval()

        W_fc1_s = W_fc1.eval()
        b_fc1_s = b_fc1.eval()
        W_fc2_s = W_fc2.eval()
        b_fc2_s = b_fc2.eval()

        save_path = saver.save(sess, "/tmp/model.ckpt")

    print("Task 1 finished. Model saved in path: %s" % save_path)

    # Once a setting has been fully trained, append its results to the task list.
    #experiment.append(
    #    ((num_epochs, learning_rate), train_accuracy, test_accuracy))


    #analysis.plot_learning_curves([experiment], log_period_samples)
    #plt.show()

    #analysis.plot_summary_table([experiment])
    #plt.show()

    # CRITICAL PERIOD MODEL
    ######################################################################
    # SETUP: Define model, loss, optimiser, training update,             #
    #        and evaluation metric (define computational graph).         #
    ######################################################################

    # Define model architecture:
    #   - two convolutional and max pooling layers, with ReLU activations
    #   - single hidden linear layer with ReLU activation
    #   - single linear layer with softmax activation
    #   - critical period initialisation of all parameters

    W_conv1crmin = W_conv1_s - constraint_half_range1*np.abs(W_conv1_s)
    W_conv1crmax = W_conv1_s + constraint_half_range1*np.abs(W_conv1_s)
    b_conv1crmin = b_conv1_s - constraint_half_range1*np.abs(b_conv1_s)
    b_conv1crmax = b_conv1_s + constraint_half_range1*np.abs(b_conv1_s)
    W_conv2crmin = W_conv2_s - constraint_half_range2*np.abs(W_conv2_s)
    W_conv2crmax = W_conv2_s + constraint_half_range2*np.abs(W_conv2_s)
    b_conv2crmin = b_conv2_s - constraint_half_range2*np.abs(b_conv2_s)
    b_conv2crmax = b_conv2_s + constraint_half_range2*np.abs(b_conv2_s)
    W_fc1crmin = W_fc1_s - constraint_half_range3*np.abs(W_fc1_s)
    W_fc1crmax = W_fc1_s + constraint_half_range3*np.abs(W_fc1_s)
    b_fc1crmin = b_fc1_s - constraint_half_range3*np.abs(b_fc1_s)
    b_fc1crmax = b_fc1_s + constraint_half_range3*np.abs(b_fc1_s)
    W_fc2crmin = W_fc2_s - constraint_half_range4*np.abs(W_fc2_s)
    W_fc2crmax = W_fc2_s + constraint_half_range4*np.abs(W_fc2_s)
    b_fc2crmin = b_fc2_s - constraint_half_range4*np.abs(b_fc2_s)
    b_fc2crmax = b_fc2_s + constraint_half_range4*np.abs(b_fc2_s)

    W_conv1cr = tf.get_variable(initializer=W_conv1_s, name='W_conv1cr', constraint=lambda W_conv1cr: \
        tf.clip_by_value(W_conv1cr, W_conv1crmin, W_conv1crmax))
    b_conv1cr = tf.get_variable(initializer=b_conv1_s, name='b_conv1cr', constraint=lambda b_conv1cr: \
        tf.clip_by_value(b_conv1cr, b_conv1crmin, b_conv1crmax))
    W_conv2cr = tf.get_variable(initializer=W_conv2_s, name='W_conv2cr', constraint=lambda W_conv2cr: \
        tf.clip_by_value(W_conv2cr, W_conv2crmin, W_conv2crmax))
    b_conv2cr = tf.get_variable(initializer=b_conv2_s, name='b_conv2cr', constraint=lambda b_conv2cr: \
        tf.clip_by_value(b_conv2cr, b_conv2crmin, b_conv2crmax))
    W_fc1cr = tf.get_variable(initializer=W_fc1_s, name='W_fc1cr',constraint=lambda W_fc1cr: \
        tf.clip_by_value(W_fc1cr, W_fc1crmin, W_fc1crmax))
    b_fc1cr = tf.get_variable(initializer=b_fc1_s, name='b_fc1cr',constraint=lambda b_fc1cr: \
        tf.clip_by_value(b_fc1cr, b_fc1crmin, b_fc1crmax))
    W_fc2cr = tf.get_variable(initializer=W_fc2_s, name='W_fc2cr',constraint=lambda W_fc2cr: \
        tf.clip_by_value(W_fc2cr, W_fc2crmin, W_fc2crmax))
    b_fc2cr = tf.get_variable(initializer=b_fc2_s, name='b_fc2cr', constraint=lambda b_fc2cr: \
        tf.clip_by_value(b_fc2cr, b_fc2crmin, b_fc2crmax))

    # CRIT PERIOD NETWORK

    h_conv1cr = tf.nn.relu(tf.nn.conv2d(x, W_conv1cr, strides=[1, 1, 1, 1],
                                        padding='SAME') + b_conv1cr)
    h_conv1cr = tf.nn.max_pool(h_conv1cr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
    h_conv1cr = tf.nn.dropout(h_conv1cr, keep_prob=prob)

    h_conv2cr = tf.nn.relu(tf.nn.conv2d(h_conv1cr, W_conv2cr, strides=[1, 1, 1, 1],
                                        padding='SAME') + b_conv2cr)
    h_conv2cr = tf.nn.max_pool(h_conv2cr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
    h_conv2cr = tf.nn.dropout(h_conv2cr, keep_prob=prob)

    # Fully-connected layers.
    h_conv2cr = tf.reshape(h_conv2cr, [-1, res // 4 * res // 4 * 8])

    h_fc1cr = tf.nn.relu(tf.matmul(h_conv2cr, W_fc1cr) + b_fc1cr)
    h_fc1cr = tf.nn.dropout(h_fc1cr, keep_prob=prob)
    y_hatcr = tf.matmul(h_fc1cr , W_fc2cr) + b_fc2cr


    cross_entropycr = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hatcr)
    total_cross_entropycr = tf.reduce_mean(cross_entropycr)

    optimizercr = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_stepcr = optimizercr.minimize(total_cross_entropycr)
    # clip_step = tf.assign(W_conv1cr, tf.clip_by_value(W_conv1cr, W_conv1crmin, W_conv1crmax))

    correct_predcr = tf.equal(tf.argmax(y, 1), tf.argmax(y_hatcr, 1))
    accuracycr = tf.reduce_mean(tf.cast(correct_predcr, dtype=tf.float32))


    print('--- Training Critical-Period Model, Task 2 ---')

    #########################################################################
    # TRAINING: Train the model (run computational graph on training data). #
    #########################################################################

    #experiment = []
    #experiment2 = []

    # Set up training session.
    i, n = 0, 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "/tmp/model.ckpt")
        print("Task 2 commenced. Model restored.")

        # Initial evaluation.
        train_accuracy2.append(sess.run(accuracycr, feed_dict=
        {x: X_train2,
         y: y_train2}))

        test_accuracy2.append(sess.run(accuracycr, feed_dict=
        {x: X_test2,
         y: y_test2}))

        while i * batch_size < num_epochs * len(X_train2):

            # Track training step number.
            i += 1
            n += 1

            # Acquire new shuffled batch.
            batch_xs = X_train2[(n - 1) * batch_size:n * batch_size, :, :, :]
            batch_ys = y_train2[(n - 1) * batch_size:n * batch_size, :]

            # Perform training step.
            sess.run(train_stepcr, feed_dict={x: batch_xs, y: batch_ys, prob: 0.8})

            # Evaluate model, if training step is a designated logging step.
            if i * batch_size % log_period_samples == 0:
                ####################################################################
                # EVALUATION: Compute and store train & test accuracy              #
                #             (run trained computational graph on evaluation data) #
                ####################################################################

                # Append training accuracy to corresponding list.
                # NOTE: the first 20% of the evaluation training dataset is arbitrarily
                #       chosen for the evaluation; for every evaluation across every
                #       model and setting, the same set is used to ensure consistency.
                train_accuracy2.append(sess.run(accuracycr, feed_dict=
                {x: X_train2,
                 y: y_train2}))
                train_accuracy1.append(sess.run(accuracycr, feed_dict=
                {x: X_train,
                 y: y_train}))

                # Append test accuracy to corrresponding list.
                test_accuracy2.append(sess.run(accuracycr, feed_dict=
                {x: X_test2,
                 y: y_test2}))
                test_accuracy1.append(sess.run(accuracycr, feed_dict=
                {x: X_test,
                 y: y_test}))

            if n % (len(X_train2) // batch_size) == 0:
                n = 0
                random_ordering_train2 = np.arange(len(X_train2))
                np.random.shuffle(random_ordering_train2)
                X_train2 = X_train2[random_ordering_train2]
                y_train2 = y_train2[random_ordering_train2]

        save_path = saver.save(sess, "/tmp/model.ckpt")

    print("Task 2 finished. Model saved in path: %s" % save_path)

    # Once a setting has been fully trained, append its results to the task list.
    #experiment.append(
    #    ((num_epochs, learning_rate), train_accuracy, test_accuracy))

    #experiment2.append(
    #    ((num_epochs, learning_rate), train_accuracy2, test_accuracy2))

    #analysis.plot_learning_curves([experiment], log_period_samples)
    #plt.show()

    #analysis.plot_summary_table([experiment])
    #plt.show()

    #analysis.plot_learning_curves([experiment2], log_period_samples)
    #plt.show()

    #analysis.plot_summary_table([experiment2])
    #plt.show()

    train_accuracies1.append(train_accuracy1)
    test_accuracies1.append(test_accuracy1)
    train_accuracies2.append(train_accuracy2)
    test_accuracies2.append(test_accuracy2)

train_accuracies1_avg = np.average(np.array(train_accuracies1), 0)
test_accuracies1_avg = np.average(np.array(test_accuracies1), 0)
train_accuracies2_avg = np.average(np.array(train_accuracies2), 0)
test_accuracies2_avg = np.average(np.array(test_accuracies2), 0)

# Save performance data of each repetition to txt files.
np.savetxt('train_accuracies1.txt', np.array(train_accuracies1))
np.savetxt('test_accuracies1.txt', np.array(test_accuracies1))
np.savetxt('train_accuracies2.txt', np.array(train_accuracies2))
np.savetxt('test_accuracies2.txt', np.array(test_accuracies2))