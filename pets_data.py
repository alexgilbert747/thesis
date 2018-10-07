import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

num_examples = 400
res = 128

def gen():

# TASK 1 DATA
    X_data = np.zeros((num_examples, res, res, 3), dtype='uint8')
    y_data = np.zeros((num_examples, 2))
    file_no = 0
    my_file = Path('')
    British_Shorthair = 'Pets/cropped_british_shorthair/British_Shorthair_'
    Russian_Blue = 'Pets/cropped_russian_blue/Russian_Blue_'
    Persian = 'Pets/cropped_persian/Persian_'
    Ragdoll = 'Pets/cropped_ragdoll/Ragdoll_'

    for i in range(0, 200):
        while my_file.is_file() == False:
            file_no += 1
            my_file = Path(
                British_Shorthair+str(file_no)+'.jpg')

        X_data[i, :, :, :] = plt.imread(
            British_Shorthair+str(file_no)+'.jpg', format='jpg')
        y_data[i, :] = np.array([1., 0.])

        my_file = Path('')

    file_no = 0
    my_file = Path('')
    for i in range(200, 400):
        while my_file.is_file() == False:
            file_no += 1
            my_file = Path(
                Persian+str(file_no)+'.jpg')

        X_data[i, :, :, :] = plt.imread(
            Persian+str(file_no)+'.jpg', format='jpg')
        y_data[i, :] = np.array([0., 1.])

        my_file = Path('')

# TASK 2 DATA
    X_data2 = np.zeros((num_examples, res, res, 3), dtype='uint8')
    y_data2 = np.zeros((num_examples, 2))
    file_no = 0
    my_file = Path('')
    for i in range(0, 200):
        while my_file.is_file() == False:
            file_no += 1
            my_file = Path(
                British_Shorthair+str(file_no)+'.jpg')

        X_data2[i, :, :, :] = plt.imread(
            British_Shorthair+str(file_no)+'.jpg', format='jpg')
        y_data2[i, :] = np.array([1., 0.])

        my_file = Path('')

    file_no = 0
    my_file = Path('')
    for i in range(200, 400):
        while my_file.is_file() == False:
            file_no += 1
            my_file = Path(
                Russian_Blue+str(file_no)+'.jpg')

        X_data2[i, :, :, :] = plt.imread(
            Russian_Blue+str(file_no)+'.jpg', format='jpg')
        y_data2[i, :] = np.array([0., 1.])

        my_file = Path('')


    return X_data, y_data, X_data2, y_data2

