import numpy as np
import matplotlib.pyplot as plt

# EXP A0
vfolder0 = 'Experiments/+expA0,vanilla,20reps,15-0.01-10-40/'
cfolder0 = 'Experiments/+expA0,critical,20reps,15-0.01-10-40-0.05-0.05/'

A0vt1 = np.average(np.loadtxt(vfolder0+'train_accuracies1.txt'), 0)
A0vt2 = np.average(np.loadtxt(vfolder0+'train_accuracies2.txt'), 0)
A0vt2 = np.concatenate((np.array([0.505412]), A0vt2))
A0v1 = np.average(np.loadtxt(vfolder0+'test_accuracies1.txt'), 0)
A0v2 = np.average(np.loadtxt(vfolder0+'test_accuracies2.txt'), 0)
A0v2 = np.concatenate((np.array([0.5012003]), A0v2))
A0ct1 = np.average(np.loadtxt(cfolder0+'train_accuracies1.txt'), 0)
A0ct2 = np.average(np.loadtxt(cfolder0+'train_accuracies2.txt'), 0)
A0c1 = np.average(np.loadtxt(cfolder0+'test_accuracies1.txt'), 0)
A0c2 = np.average(np.loadtxt(cfolder0+'test_accuracies2.txt'), 0)

# EXP A1
vfolder1 = 'Experiments/+expA1,vanilla,20reps,100-0.01-120-960/'
cfolder1 = 'Experiments/+expA1,critical,20reps,100-0.01-120-960-0.05-0.20/'

A1vt1 = np.average(np.loadtxt(vfolder1+'train_accuracies1.txt'), 0)
A1vt2 = np.average(np.loadtxt(vfolder1+'train_accuracies2.txt'), 0)
#A1vt2 = np.concatenate((np.array([0.496912]), A1vt2))
A1v1 = np.average(np.loadtxt(vfolder1+'test_accuracies1.txt'), 0)
A1v2 = np.average(np.loadtxt(vfolder1+'test_accuracies2.txt'), 0)
#A1v2 = np.concatenate((np.array([0.5038307]), A1v2))
A1ct1 = np.average(np.loadtxt(cfolder1+'train_accuracies1.txt'), 0)
A1ct2 = np.average(np.loadtxt(cfolder1+'train_accuracies2.txt'), 0)
A1c1 = np.average(np.loadtxt(cfolder1+'test_accuracies1.txt'), 0)
A1c2 = np.average(np.loadtxt(cfolder1+'test_accuracies2.txt'), 0)

# EXP A2
vfolder2 = 'Experiments/+expA2,vanilla,20reps,100-0.01-120-960/'
cfolder2 = 'Experiments/+expA2,critical,20reps,100-0.01-120-960-/'

A2vt1 = np.average(np.loadtxt(vfolder2+'train_accuracies1.txt'), 0)
A2vt2 = np.average(np.loadtxt(vfolder2+'train_accuracies2.txt'), 0)
A2v1 = np.average(np.loadtxt(vfolder2+'test_accuracies1.txt'), 0)
A2v2 = np.average(np.loadtxt(vfolder2+'test_accuracies2.txt'), 0)
A2ct1 = np.average(np.loadtxt(cfolder2+'train_accuracies1.txt'), 0)
A2ct2 = np.average(np.loadtxt(cfolder2+'train_accuracies2.txt'), 0)
A2c1 = np.average(np.loadtxt(cfolder2+'test_accuracies1.txt'), 0)
A2c2 = np.average(np.loadtxt(cfolder2+'test_accuracies2.txt'), 0)

# EXP A4
vfolder4 = 'Experiments/+expA4,vanilla,20reps,40-0.01-40-960/'
cfolder4 = 'Experiments/+expA4,critical,20reps,40-0.01-40-960-0.05-0.05/'

A4vt1 = np.average(np.loadtxt(vfolder4+'train_accuracies1.txt'), 0)
A4vt2 = np.average(np.loadtxt(vfolder4+'train_accuracies2.txt'), 0)
A4v1 = np.average(np.loadtxt(vfolder4+'test_accuracies1.txt'), 0)
A4v2 = np.average(np.loadtxt(vfolder4+'test_accuracies2.txt'), 0)
A4ct1 = np.average(np.loadtxt(cfolder4+'train_accuracies1.txt'), 0)
A4ct2 = np.average(np.loadtxt(cfolder4+'train_accuracies2.txt'), 0)
A4c1 = np.average(np.loadtxt(cfolder4+'test_accuracies1.txt'), 0)
A4c2 = np.average(np.loadtxt(cfolder4+'test_accuracies2.txt'), 0)

# EXP A5
#vfolder = 'bleb/'
#cfolder = 'bleb2/'

vfolder5 = 'Experiments/+expA5,vanilla,20rep,40-0.01-40-960/'
cfolder5 = 'Experiments/+expA5,critical,20rep,40-0.01-40-960-final/'

#expA5,critical,1rep,40-0.01-40-960-increasing2/
#expA5,critical,1rep,40-0.01-40-960-increasing1/
'''
A5vt1 = np.loadtxt(vfolder+'train_accuracies1.txt')
A5vt2 = np.loadtxt(vfolder+'train_accuracies2.txt')
A5vt3 = np.loadtxt(vfolder+'train_accuracies3.txt')
A5vt4 = np.loadtxt(vfolder+'train_accuracies4.txt')
A5v1 = np.loadtxt(vfolder+'test_accuracies1.txt')
A5v2 = np.loadtxt(vfolder+'test_accuracies2.txt')
A5v3 = np.loadtxt(vfolder+'test_accuracies3.txt')
A5v4 = np.loadtxt(vfolder+'test_accuracies4.txt')

A5ct1 = np.loadtxt(cfolder+'train_accuracies1.txt')
A5ct2 = np.loadtxt(cfolder+'train_accuracies2.txt')
A5ct3 = np.loadtxt(cfolder+'train_accuracies3.txt')
A5ct4 = np.loadtxt(cfolder+'train_accuracies4.txt')
A5c1 = np.loadtxt(cfolder+'test_accuracies1.txt')
A5c2 = np.loadtxt(cfolder+'test_accuracies2.txt')
A5c3 = np.loadtxt(cfolder+'test_accuracies3.txt')
A5c4 = np.loadtxt(cfolder+'test_accuracies4.txt')
'''
A5vt1 = np.average(np.loadtxt(vfolder5+'train_accuracies1.txt'), 0)
A5vt2 = np.average(np.loadtxt(vfolder5+'train_accuracies2.txt'), 0)
A5vt3 = np.average(np.loadtxt(vfolder5+'train_accuracies3.txt'), 0)
A5vt4 = np.average(np.loadtxt(vfolder5+'train_accuracies4.txt'), 0)
A5v1 = np.average(np.loadtxt(vfolder5+'test_accuracies1.txt'), 0)
A5v2 = np.average(np.loadtxt(vfolder5+'test_accuracies2.txt'), 0)
A5v3 = np.average(np.loadtxt(vfolder5+'test_accuracies3.txt'), 0)
A5v4 = np.average(np.loadtxt(vfolder5+'test_accuracies4.txt'), 0)

A5ct1 = np.average(np.loadtxt(cfolder5+'train_accuracies1.txt'), 0)
A5ct2 = np.average(np.loadtxt(cfolder5+'train_accuracies2.txt'), 0)
A5ct3 = np.average(np.loadtxt(cfolder5+'train_accuracies3.txt'), 0)
A5ct4 = np.average(np.loadtxt(cfolder5+'train_accuracies4.txt'), 0)
A5c1 = np.average(np.loadtxt(cfolder5+'test_accuracies1.txt'), 0)
A5c2 = np.average(np.loadtxt(cfolder5+'test_accuracies2.txt'), 0)
A5c3 = np.average(np.loadtxt(cfolder5+'test_accuracies3.txt'), 0)
A5c4 = np.average(np.loadtxt(cfolder5+'test_accuracies4.txt'), 0)

# EXP A7
vfolder7 = 'Experiments/+expA7,vanilla,20rep,40-0.01-40-960/'
cfolder7 = 'Experiments/+expA7,critical,20rep,40-0.01-40-960-final/'

A7vt1 = np.average(np.loadtxt(vfolder7+'train_accuracies1.txt'), 0)
A7vt2 = np.average(np.loadtxt(vfolder7+'train_accuracies2.txt'), 0)
A7vt3 = np.average(np.loadtxt(vfolder7+'train_accuracies3.txt'), 0)
A7vt4 = np.average(np.loadtxt(vfolder7+'train_accuracies4.txt'), 0)
A7v1 = np.average(np.loadtxt(vfolder7+'test_accuracies1.txt'), 0)
A7v2 = np.average(np.loadtxt(vfolder7+'test_accuracies2.txt'), 0)
A7v3 = np.average(np.loadtxt(vfolder7+'test_accuracies3.txt'), 0)
A7v4 = np.average(np.loadtxt(vfolder7+'test_accuracies4.txt'), 0)

A7ct1 = np.average(np.loadtxt(cfolder7+'train_accuracies1.txt'), 0)
A7ct2 = np.average(np.loadtxt(cfolder7+'train_accuracies2.txt'), 0)
A7ct3 = np.average(np.loadtxt(cfolder7+'train_accuracies3.txt'), 0)
A7ct4 = np.average(np.loadtxt(cfolder7+'train_accuracies4.txt'), 0)
A7c1 = np.average(np.loadtxt(cfolder7+'test_accuracies1.txt'), 0)
A7c2 = np.average(np.loadtxt(cfolder7+'test_accuracies2.txt'), 0)
A7c3 = np.average(np.loadtxt(cfolder7+'test_accuracies3.txt'), 0)
A7c4 = np.average(np.loadtxt(cfolder7+'test_accuracies4.txt'), 0)