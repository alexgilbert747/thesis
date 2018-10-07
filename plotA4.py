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

x1 = np.arange(0, 81)
x2 = np.arange(40, 81)


divider1y = np.array([0.0, 1.0])
divider1x = np.array([40, 40])
randomy = np.array([0.5, 0.5])
randomx = np.array([0, 80])


ax = plt.subplot()

#######################################################
plt.plot(divider1x, divider1y, color='black', linestyle=':', linewidth=0.725)
plt.plot(randomx, randomy, color='black', linestyle='-.', linewidth=0.725)
#ax.plot(x1, A4ct1, color=mid_purple, linewidth=0.725, alpha=0.25)
#ax.plot(x1, A4vt1, color=orange, linewidth=0.725, alpha=0.25)
plt.plot(x1, A4c1, color='#1f77b4', linewidth=0.725)
plt.plot(x1, A4v1, color='#1f77b4', linewidth=0.725, linestyle='--')
plt.plot(x2, A4c2, color='#ff7f0e', linewidth=0.725)
plt.plot(x2, A4v2, color='#ff7f0e', linewidth=0.725, linestyle='--')


ax.text(20, 0.45, 'Task 1', color='#1f77b4', verticalalignment='center', horizontalalignment='center')
ax.text(60, 0.45, 'Task 2', color='#ff7f0e', verticalalignment='center', horizontalalignment='center')

ax.text(80, 0.495, 'Random', verticalalignment='center', horizontalalignment='right')
ax.text(80, A4c1[-1], 'Critical (Task 1)', verticalalignment='center', horizontalalignment='right')
ax.text(80, A4c2[-1], 'Critical (Task 2)', verticalalignment='center', horizontalalignment='right')

ax.text(80, A4v1[-1], 'Reference (Task 1)', verticalalignment='center', horizontalalignment='right')
ax.text(80, A4v2[-1], 'Reference (Task 2)', verticalalignment='center', horizontalalignment='right')


#horizontalalignment='center',
#...      verticalalignment='center'
#######################################################
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(right=0.75)

#ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel('Training time')
plt.ylabel('Test accuracy')

plt.ylim(0.45, 1.0)
plt.xlim(0, 80)

#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.tick_params(
    axis='y',
    right=True,
    labelright=True)

print('vanilla:',(A4v1[-1] + A4v2[-1])/2,'+/-',np.std(np.array([A4v1[-1], A4v2[-1]])))
print('critical:',(A4c1[-1] + A4c2[-1])/2,'+/-',np.std(np.array([A4c1[-1], A4c2[-1]])))

tikz_save('Graphs/plotA4.tikz', figureheight='8cm', figurewidth='13.69785cm')