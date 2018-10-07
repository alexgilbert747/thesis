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

x1 = 1
x2 = 2
x3 = 3
x4 = 4


ax = plt.subplot()

#######################################################
plt.bar(x1, A5c1[40], width=1, color='#1f77b4', alpha=0.3)#bottom=A7c1[-1])
plt.bar(x2, A5c2[40], width=1, color='#ff7f0e', alpha=0.3)#bottom=A7c2[-1])
plt.bar(x3, A5c3[40], width=1, color='#2ca02c', alpha=0.3)#bottom=A7c3[-1])
plt.bar(x4, A5c4[40], width=1, color='#d62728', alpha=0.3)#bottom=A7c4[-1])

plt.bar(x1, A5c1[-1], width=1, color='#1f77b4')
plt.bar(x2, A5c2[-1], width=1, color='#ff7f0e')
plt.bar(x3, A5c3[-1], width=1, color='#2ca02c')
plt.bar(x4, A5c4[-1], width=1, color='#d62728')

#plt.plot(x1, A7c1, color='#1f77b4', linewidth=0.725)
#plt.plot(x1, A7v1, color='#1f77b4', linewidth=0.725, linestyle='--')
#plt.plot(x2, A7c2, color='#ff7f0e', linewidth=0.725)
#plt.plot(x2, A7v2, color='#ff7f0e', linewidth=0.725, linestyle='--')
#plt.plot(x3, A7c3, color='#2ca02c', linewidth=0.725)
#plt.plot(x3, A7v3, color='#2ca02c', linewidth=0.725, linestyle='--')
#plt.plot(x4, A7c4, color='#d62728', linewidth=0.725)
#plt.plot(x4, A7v4, color='#d62728', linewidth=0.725, linestyle='--')

'''
ax.text(20, 0.05, 'Task 1', color='#1f77b4', verticalalignment='center', horizontalalignment='center')
ax.text(60, 0.05, 'Task 2', color='#ff7f0e', verticalalignment='center', horizontalalignment='center')
ax.text(100, 0.05, 'Task 3', color='#2ca02c', verticalalignment='center', horizontalalignment='center')
ax.text(140, 0.05, 'Task 4', color='#d62728', verticalalignment='center', horizontalalignment='center')

ax.text(160, 0.495, 'Random', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7c1[-1], 'Critical (Task 1)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7c2[-1]+0.01, 'Critical (Task 2)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7c3[-1]+0.01, 'Critical (Task 3)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7c4[-1], 'Critical (Task 4)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7v1[-1], 'Reference (Task 1)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7v2[-1], 'Reference (Task 2)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7v3[-1]-0.01, 'Reference (Task 3)', verticalalignment='center', horizontalalignment='right')
ax.text(160, A7v4[-1], 'Reference (Task 4)', verticalalignment='center', horizontalalignment='right')
'''

#######################################################
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(right=0.75)

#ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel('')
plt.ylabel('Test accuracy')

plt.ylim(0.0, 1.0)
plt.xlim(0.5, 4.5)

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

print('vanilla:',(A5v1[-1] + A5v2[-1] + A5v3[-1] + A5v4[-1])/4,'+/-',np.std(np.array([A5v1[-1], A5v2[-1], A5v3[-1], A5v4[-1]])))
print('critical:',(A5c1[-1] + A5c2[-1] + A5c3[-1] + A5c4[-1])/4,'+/-',np.std(np.array([A5c1[-1], A5c2[-1], A5c3[-1], A5c4[-1]])))

tikz_save('Graphs/barA5.tikz', figureheight='8cm', figurewidth='13.69785cm')