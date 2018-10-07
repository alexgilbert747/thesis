import numpy as np
import matplotlib.pyplot as plt

# Set size of squares and circles.
d = 31#61
r = int(d/2)

# Set linear dimension of square canvas.
res = 64 #128
pad = 2 # boundary number of pixels

# Set number of data examples to be generated.
num_examples = 1000


def gen():
    # Create square stamp.
    square = np.zeros((d,d,3))
    square[:,:,0:1] = 1

    # Create circle stamp.
    circle = np.zeros((d,d,3))
    for y in range(np.shape(circle)[0]):
        for x in range(np.shape(circle)[1]):
            if np.sqrt((x-r)*(x-r) + (r-y)*(r-y)) < r:
                circle[y,x,0:1] = 1

    # Create triangle stamp.
    triangle = np.zeros((d,d,3))
    for y in range(np.shape(triangle)[0]):
        for x in range(np.shape(triangle)[1]):
            if ((d//2)-y) < -(x-(d//2)) and (x-(d//2)) > ((d//2)-y):
                triangle[y,x,0:1] = 1

    # Create data array of blank input canvases and blank labels.
    X_data = np.zeros((num_examples, res, res, 3))
    y_data = np.zeros((num_examples, 2))
    '''
    X_data = np.zeros((num_examples, res, res, 3))
    y_data = np.zeros((num_examples, 2))
    '''
    # Randomly stamp a circle or square somewhere on each canvas and label appropriately.
    for i in range(num_examples):
        rand_o = np.random.randint(0 + pad, res - d - pad, 2)
        if np.random.randint(0, 2) == 0:

            X_data[i, rand_o[0]:rand_o[0]+d, rand_o[1]:rand_o[1]+d, :] = triangle
            y_data[i, :] = np.array([1, 0])
            '''
            X_data[i, :, :, :] = circle
            y_data[i,:] = np.array([1,0])
            '''
        else:

            X_data[i, rand_o[0]:rand_o[0] + d, rand_o[1]:rand_o[1] + d, :] = square
            y_data[i, :] = np.array([0, 1])
            '''
            X_data[i, :, :, :] = square
            y_data[i,:] = np.array([0,1])
            '''

    # Split data and label arrays into training and test sets.
    X_train = X_data[0:int(0.8*num_examples)]
    y_train = y_data[0:int(0.8*num_examples)]
    X_test = X_data[int(0.8*num_examples):]
    y_test = y_data[int(0.8*num_examples):]

    return X_train, y_train, X_test, y_test
    #imgplot = plt.imshow(example)
    #plt.axis('off')
    #plt.show(imgplot)