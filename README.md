# thesis
Lifelong machine learning inspired by neocortical critical-period plasticity

README FOR THE 'LIFELONG MACHINE LEARNING INSPIRED BY NEOCORTICAL CRITICAL-PERIOD PLASTICITY' RESEARCH PROJECT BY A.J. GILBERT.

There are EXPERIMENT A CODE FILES, EXPERIMENT B CODE FILES, and FOLDERS of results etc. provided in this repository.

EXPERIMENT A CODE FILES:

"pets_data2.py" imports images for a sequence of 2 binary classification tasks (user specified cat species)

"pets_data4.py" imports images for a sequence of 4 binary classification tasks (user specified cat species)

"experiment1.py" trains a reference network for a sequence of 2 binary classification tasks.

"experiment1b.py" trains a critical-period network for a sequence of 2 binary classification tasks.

"experiment1m.py" trains a reference network for a sequence of 4 binary classification tasks.
"experiment1bm.py" trains a critical-period network for a sequence of 4 binary classification tasks.
"visualise.py" defines functions for visualising hidden activations.
"experiment1_visualisations.py" trains and then produces visualisations for a reference network for a sequence of 2 binary classification tasks.

"plot_import_data" imports all key results from the 'Experiments' folder and allows for plotting the graphs using:
"plotA0.py" plots the test accuracies for A0
"plotA1.py" plots the test accuracies for A1
"plotA2.py" plots the test accuracies for A2
"plotA4.py" plots the test accuracies for A4
"plotA5.py" plots the test accuracies for A5
"plotA7.py" plots the test accuracies for A7
"plotper.py" plots the A7 mean test accuracies for increasing numbers of tasks.
"barA4.py" plots bar charts for the test accuracies per task for A4.
"barA5.py" plots bar charts for the test accuracies per task for A5.
"barA7.py" plots bar charts for the test accuracies per task for A7.


EXPERIMENT B CODE FILES:

"auxiliary.py" are auxiliary functions, e.g. activation functions.
"analysis.py" are analysis functions.

"environment.py" defines the noughts-and-crosses environment.
"agent_tabular.py" defines a tabular SARSA agent.
"agent_neural.py" defines a 2-layer SARSA agent.
"agent_neural2.py" defines a 3-layer SARSA agent.
"experiment2.py" trains the tabular SARSA agent.
"experiment3.py" trains the 2-layer SARSA agent.
"experiment4.py" trains the 4-layer SARSA agent.

FOLDERS:

- Folder "Experiments" contains text files of all results obtained. These can be loaded into NumPy arrays using:

e.g.
np.loadtxt('Experiments/expA4,vanilla,20reps,40-0.01-40-960/train_accuracies1.txt')

sub-folder naming convention:

e.g.
'expA4,vanilla,20reps,40-0.01-40-960' means Run A4, Vanilla (reference) network, 20 repetitions, 40 epochs, 0.01 learning rate, 40 batch size, 960 examples processed for each accuracy evaluation (i.e. accuracy evaluated every: 960/batch_size training steps).

- Folder "Graphs" contains .tikz files of all graphs obtained from the code.

- Folder "Pets" contains the pre-processed images. 

sub-folders beginning with "crop64" contain the images that were finally used in the experiment.

- Folder "Visualisations" contains visualisations of the hidden activations of the convolutional layers.

