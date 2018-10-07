import matplotlib.pyplot as plt
from auxiliary import *
import math


def plot_filters(cat, folder, activations):
    filters = activations.shape[3]
    #plt.figure(fig, figsize=(20, 20))
    #n_columns = 6
    #n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        #plt.subplot(n_rows, n_columns, i + 1)
        #plt.imshow(activations[0, :, :, i], interpolation="nearest") #, cmap="gist_rainbow"
        #plt.show()
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=activations[0, :, :, i].min(), vmax=activations[0, :, :, i].max())
        image = cmap(norm(activations[0, :, :, i]))
        plt.imsave('Visualisations/' + folder + cat + str(i) + '.png', image)


def getActivations(sess, prob, x, layer, test_img):
    activations = sess.run(layer, feed_dict={x: np.reshape(test_img, [1, 64, 64, 3], order='F'), prob: 1.0})
    return activations
