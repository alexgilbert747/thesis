# PET DATA PROCESSING

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

num_examples = 1388
num_examples2 = 1396
num_examples3 = 1402
num_examples4 = 1392
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
    X_data3 = np.zeros((num_examples3, res, res, 3), dtype='uint8')
    y_data3 = np.zeros((num_examples3, 2))
    X_data4 = np.zeros((num_examples4, res, res, 3), dtype='uint8')
    y_data4 = np.zeros((num_examples4, 2))

    British_Shorthair = 'Pets/crop64_british_shorthair/'
    Bengal = 'Pets/crop64_bengal/'
    Siamese = 'Pets/crop64_siamese/'
    Persian = 'Pets/crop64_persian/'
    Maine_Coon = 'Pets/crop64_maine_coon/'
    Russian_Blue = 'Pets/crop64_russian_blue/'
    Sphynx = 'Pets/crop64_sphynx/'
    Ragdoll = 'Pets/crop64_ragdoll/'
    Abyssinian = 'Pets/crop64_abyssinian/'
    Bombay = 'Pets/crop64_bombay/'

    # TASK 1 DATA
    folder_to_array(British_Shorthair, 'British_Shorthair_', X_data, 0, 200, 147, y_data, np.array([1., 0.]))
    folder_to_array(Bengal, 'Bengal_', X_data, 694, 200, 147, y_data, np.array([0., 1.]))
    # brit vs bombay
    # TASK 2 DATA
    folder_to_array(Siamese, 'Siamese_', X_data2, 0, 200, 146, y_data2, np.array([1., 0.]))
    folder_to_array(Persian, 'Persian_', X_data2, 692, 200, 152, y_data2, np.array([0., 1.]))
    # ragd vs persia
    # TASK 3 DATA
    folder_to_array(Sphynx, 'Sphynx_', X_data3, 0, 200, 148, y_data3, np.array([1., 0.]))
    folder_to_array(Ragdoll, 'Ragdoll_', X_data3, 696, 200, 153, y_data3, np.array([0., 1.]))
    # sphy vs beng
    # TASK 4 DATA
    folder_to_array(Maine_Coon, 'Maine_Coon_', X_data4, 0, 200, 144, y_data4, np.array([1., 0.]))
    folder_to_array(Bombay, 'Bombay_', X_data4, 688, 200, 152, y_data4, np.array([0., 1.]))
    # coon/abys vs siam
    '''
    # TASK 1 DATA
    folder_to_array(British_Shorthair, 'British_Shorthair_', X_data, 0, 200, 147, y_data, np.array([1., 0.]))
    folder_to_array(Bengal, 'Bengal_', X_data, 694, 200, 147, y_data, np.array([0., 1.]))

    # TASK 2 DATA
    folder_to_array(Siamese, 'Siamese_', X_data2, 0, 200, 146, y_data2, np.array([1., 0.]))
    folder_to_array(Persian, 'Persian_', X_data2, 688, 200, 152, y_data2, np.array([0., 1.]))

    # TASK 3 DATA
    folder_to_array(Sphynx, 'Sphynx_', X_data3, 0, 200, 148, y_data3, np.array([1., 0.]))
    folder_to_array(Ragdoll, 'Ragdoll_', X_data3, 696, 200, 153, y_data3, np.array([0., 1.]))

    # TASK 4 DATA
    folder_to_array(Maine_Coon, 'Maine_Coon_', X_data4, 0, 200, 144, y_data4, np.array([1., 0.]))
    folder_to_array(Russian_Blue, 'Russian_Blue_', X_data4, 688, 200, 144, y_data4, np.array([0., 1.]))

    '''

    return X_data, y_data, X_data2, y_data2, X_data3, y_data3, X_data4, y_data4