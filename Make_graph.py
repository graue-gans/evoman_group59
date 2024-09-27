from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from deap.gp import generate

# Load data from logs
dir_name = 'optimization_test'
enemy_number = '8'
data_2px = np.load('cx_' + enemy_number + '/logs1.npz')
data_blendx = np.load('blend_' + enemy_number + '/logs1.npz')


# Create generations vector
generations = np.arange(1, len(data_2px['max_fitness']) + 1)

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 10})

# Plot convergence threshold line at fitness = 80
plt.plot([-5, max(generations)+5], [80, 80], 'k:', alpha=0.4, label='Convergence threshold')

# Plot EA1 max fitness
plt.plot(generations, data_2px['max_fitness'], label='EA1: Max Fitness', color='blue', linestyle='-')
plt.fill_between(generations, data_2px['max_fitness'] - data_2px['std_fitness'],
                 data_2px['max_fitness'] + data_2px['std_fitness'], color='blue', alpha=0.2)

# Plot EA1 average fitness
plt.plot(generations, data_2px['avg_fitness'], label='EA1: Avg Fitness', color='purple', linestyle='--')
plt.fill_between(generations, data_2px['avg_fitness'] - data_2px['std_avg_fitness'],
                 data_2px['avg_fitness'] + data_2px['std_avg_fitness'], color='purple', alpha=0.2)

# Plot EA2 max fitness
plt.plot(generations, data_blendx['max_fitness'], label='EA2: Max Fitness', color='green', linestyle='-')
plt.fill_between(generations, data_blendx['max_fitness'] - data_blendx['std_fitness'],
                 data_blendx['max_fitness'] + data_blendx['std_fitness'], color='green', alpha=0.2)

# Plot EA2 average fitness
plt.plot(generations, data_blendx['avg_fitness'], label='EA2: Avg Fitness', color='orange', linestyle='--')
plt.fill_between(generations, data_blendx['avg_fitness'] - data_blendx['std_avg_fitness'],
                 data_blendx['avg_fitness'] + data_blendx['std_avg_fitness'], color='orange', alpha=0.2)

# Add labels and legend (title is added in latex)
plt.xlabel('Generation')
plt.ylabel('Fitness Metrics')
# plt.title('Comparison of Fitness Metrics with Shaded Standard Deviation for EA1 and EA2')
plt.legend(loc='best', fontsize=8)

# Edit x-limits and ticks
plt.xlim((0, max(generations)+1))
plt.xticks(np.arange(0, max(generations)+1, step=2))

plt.tight_layout()
plt.show()
