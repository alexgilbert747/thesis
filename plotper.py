import numpy as np
import matplotlib.pyplot as plt
from plot_import_data import *
from matplotlib2tikz import save as tikz_save

mid_purple = '#500778'
mid_red = '#93272C'
bright_red = '#D50032'
bright_green = '#B5BD00'
orange = '#EA7600'
blu = '#A4DBE8'
purple = '#9467bd'
pink = '#e377c2'

x = np.arange(1, 5)

ax = plt.subplot()

#######################################################
lineMv = np.loadtxt('Experiments/per_num_tasks/pernumtasksv.txt')
lineMc = np.loadtxt('Experiments/per_num_tasks/pernumtasksc.txt')
singley = np.array([0.85, 0.85])
singlex = np.array([1, 5])

plt.plot(singlex, singley, color=pink, linewidth=0.725)
plt.plot(x, lineMc, color=purple, linewidth=0.725)
plt.plot(x, lineMv, color=purple, linewidth=0.725, linestyle='--')



ax.text(4, 0.85, 'Single-task performance', verticalalignment='center', horizontalalignment='right')
ax.text(4, lineMc[-1], 'Critical', verticalalignment='center', horizontalalignment='right')
ax.text(4, lineMv[-1], 'Reference', verticalalignment='center', horizontalalignment='right')

#horizontalalignment='center',
#...      verticalalignment='center'
#######################################################
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)


#ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel('Number of tasks')
plt.ylabel('Mean test accuracy')

plt.ylim(0, 1.0)
plt.xlim(1, 4)

plt.xticks([1,2,3,4])

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True)
plt.tick_params(
    axis='y',
    right=True,
    labelright=True)

print('vanilla:',(A4v1[-1] + A4v2[-1])/2,'+/-',np.std(np.array([A4v1[-1], A4v2[-1]])))
print('critical:',(A4c1[-1] + A4c2[-1])/2,'+/-',np.std(np.array([A4c1[-1], A4c2[-1]])))

tikz_save('Graphs/plotper.tikz', figureheight='8cm', figurewidth='13.69785cm')