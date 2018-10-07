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

x1 = np.arange(0, 161)
x2 = np.arange(40, 161)
x3 = np.arange(80, 161)
x4 = np.arange(120, 161)

divider1y = np.array([0.0, 1.0])
divider1x = np.array([40, 40])
divider2y = np.array([0.0, 1.0])
divider2x = np.array([80, 80])
divider3y = np.array([0.0, 1.0])
divider3x = np.array([120, 120])
randomy = np.array([0.5, 0.5])
randomx = np.array([0, 160])


ax = plt.subplot()

#######################################################
plt.plot(divider1x, divider1y, color='black', linestyle=':', linewidth=0.725)
plt.plot(divider2x, divider2y, color='black', linestyle=':', linewidth=0.725)
plt.plot(divider3x, divider3y, color='black', linestyle=':', linewidth=0.725)
plt.plot(randomx, randomy, color='black', linestyle='-.', linewidth=0.725)
#ax.plot(x1, A4ct1, color=mid_purple, linewidth=0.725, alpha=0.25)
#ax.plot(x1, A4vt1, color=orange, linewidth=0.725, alpha=0.25)
plt.plot(x1, A5c1, color='#1f77b4', linewidth=0.725)
plt.plot(x1, A5v1, color='#1f77b4', linewidth=0.725, linestyle='--')
plt.plot(x2, A5c2, color='#ff7f0e', linewidth=0.725)
plt.plot(x2, A5v2, color='#ff7f0e', linewidth=0.725, linestyle='--')
plt.plot(x3, A5c3, color='#2ca02c', linewidth=0.725)
plt.plot(x3, A5v3, color='#2ca02c', linewidth=0.725, linestyle='--')
plt.plot(x4, A5c4, color='#d62728', linewidth=0.725)
plt.plot(x4, A5v4, color='#d62728', linewidth=0.725, linestyle='--')

'''
ax.annotate('critical (task 1)', xy=(160, A5c1[-1]), xytext=(180, A5c1[-1]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.725, headwidth=6))
ax.annotate('critical (task 2)', xy=(160, A5c2[-1]), xytext=(180, A5c2[-1]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.725, headwidth=6))
ax.annotate('critical (task 3)', xy=(160, A5c3[-1]), xytext=(180, A5c3[-1]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.725, headwidth=6))
ax.annotate('critical (task 4)', xy=(160, A5c4[-1]), xytext=(180, A5c4[-1]),
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.725, headwidth=6))
'''

ax.text(20, 0.05, 'Task 1', color='#1f77b4', verticalalignment='center', horizontalalignment='center')
ax.text(60, 0.05, 'Task 2', color='#ff7f0e', verticalalignment='center', horizontalalignment='center')
ax.text(100, 0.05, 'Task 3', color='#2ca02c', verticalalignment='center', horizontalalignment='center')
ax.text(140, 0.05, 'Task 4', color='#d62728', verticalalignment='center', horizontalalignment='center')

ax.text(140, 0.495, 'Random', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5c1[-1], 'Critical (Task 1)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5c2[-1]+0.01, 'Critical (Task 2)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5c3[-1]+0.01, 'Critical (Task 3)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5c4[-1], 'Critical (Task 4)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5v1[-1], 'Reference (Task 1)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5v2[-1], 'Reference (Task 2)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5v3[-1]-0.01, 'Reference (Task 3)', verticalalignment='center', horizontalalignment='center')
ax.text(140, A5v4[-1], 'Reference (Task 4)', verticalalignment='center', horizontalalignment='center')

#horizontalalignment='center',
#...      verticalalignment='center'
#######################################################
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.subplots_adjust(right=0.75)

#ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlabel('Training time')
plt.ylabel('Fraction correct')

plt.ylim(0.0, 1.0)
plt.xlim(0, 160)

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

tikz_save('Graphs/plotA5b.tikz', figureheight='8cm', figurewidth='14.69785cm')