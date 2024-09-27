import numpy as np
import matplotlib.pyplot as plt

dir_name = 'optimization_test'

data_EA1 = np.load(dir_name +'/logs.npz')
data_EA2 = np.load(dir_name +'/logs1.npz')



# Dynamically define x-axis based on the array lengths
generations_EA1 = np.arange(1, len(data_EA1['max_fitness']) + 1)
generations_EA2 = np.arange(1, len(data_EA2['max_fitness']) + 1)

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 10})

plt.plot(generations_EA1, data_EA1['max_fitness'], label='EA1: Max Fitness', color='blue', linestyle='-')
plt.fill_between(generations_EA1, data_EA1['max_fitness'] - data_EA1['std_fitness'],
                 data_EA1['max_fitness'] + data_EA1['std_fitness'], color='blue', alpha=0.2)

plt.plot(generations_EA2, data_EA2['max_fitness'], label='EA2: Max Fitness', color='green', linestyle='--')
plt.fill_between(generations_EA2, data_EA2['max_fitness'] - data_EA2['std_fitness'],
                 data_EA2['max_fitness'] + data_EA2['std_fitness'], color='green', alpha=0.2)

plt.plot(generations_EA1, data_EA1['avg_fitness'], label='EA1: Avg Fitness', color='purple', linestyle='-')
plt.fill_between(generations_EA1, data_EA1['avg_fitness'] - data_EA1['std_avg_fitness'],
                 data_EA1['avg_fitness'] + data_EA1['std_avg_fitness'], color='purple', alpha=0.2)

plt.plot(generations_EA2, data_EA2['avg_fitness'], label='EA2: Avg Fitness', color='orange', linestyle='--')
plt.fill_between(generations_EA2, data_EA2['avg_fitness'] - data_EA2['std_avg_fitness'],
                 data_EA2['avg_fitness'] + data_EA2['std_avg_fitness'], color='orange', alpha=0.2)

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness Metrics')
plt.title('Comparison of Fitness Metrics with Shaded Standard Deviation for EA1 and EA2')
plt.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()