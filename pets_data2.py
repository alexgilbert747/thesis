# PET DATA PROCESSING

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

num_examples = 1386
num_examples2 = 1386
res = 64

def folder_to_array(file_name, cat_name, X, idx1, numf, numt, y, label):
    idx_normal = range(idx1, idx1+numf)
    idx_flip   = range(idx1+numf, idx1+2*numf)
    idx_twist  = range(idx1+2*numf, idx1+2*numf+numt)
    idx_twistflip = range(idx1+2*numf+numt, idx1+2*numf+2*numt)
    modes = ['','flip/','twist/','twistflip/']
    for m, idx_range in enumerate([idx_normal, idx_flip, idx_twist, idx_twistflip]):
        file_no = 0
        my_file = Path('')
        for i in idx_range:
            while my_file.is_file() == False:
                file_no += 1
                my_file = Path(file_name+modes[m]+cat_name+str(file_no)+'.jpg')

            X[i, :, :, :] = plt.imread(file_name+modes[m]+cat_name+str(file_no)+'.jpg', format='jpg')
            y[i, :] = label

            my_file = Path('')

def gen():

    X_data = np.zeros((num_examples, res, res, 3), dtype='uint8')
    y_data = np.zeros((num_examples, 2))
    X_data2 = np.zeros((num_examples2, res, res, 3), dtype='uint8')
    y_data2 = np.zeros((num_examples2, 2))

    British_Shorthair = 'Pets/crop64_british_shorthair/'
    Siamese = 'Pets/crop64_siamese/'
    Persian = 'Pets/crop64_persian/'
    Ragdoll = 'Pets/crop64_ragdoll/'
    Bengal = 'Pets/crop64_bengal/'
    Bombay = 'Pets/crop64_bombay/'

    # TASK 1 DATA
    folder_to_array(British_Shorthair, 'British_Shorthair_', X_data, 0, 200, 147, y_data, np.array([1., 0.]))
    folder_to_array(Siamese, 'Siamese_', X_data, 694, 200, 146, y_data, np.array([0., 1.]))

    # TASK 2 DATA
    folder_to_array(Siamese, 'Siamese_', X_data2, 0, 200, 146, y_data2, np.array([1., 0.]))
    folder_to_array(British_Shorthair, 'British_Shorthair_', X_data2, 692, 200, 147, y_data2, np.array([0., 1.]))

    return X_data, y_data, X_data2, y_data2

