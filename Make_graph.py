import numpy as np
import matplotlib.pyplot as plt

dir_name = 'optimization_test'

data_EA1 = np.load(dir_name +'/logs.npz')
data_EA2 = np.load(dir_name +'/logs1.npz')



generations_EA1 = np.arange(1, len(data_EA1['max_fitness']) + 1)
generations_EA2 = np.arange(1, len(data_EA2['max_fitness']) + 1)

max_fitness_value_EA1 = np.max(data_EA1['max_fitness'])
max_fitness_gen_EA1 = generations_EA1[np.argmax(data_EA1['max_fitness'])]

max_fitness_value_EA2 = np.max(data_EA2['max_fitness'])
max_fitness_gen_EA2 = generations_EA2[np.argmax(data_EA2['max_fitness'])]

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 10})

plt.errorbar(generations_EA1, data_EA1['max_fitness'], yerr=data_EA1['std_fitness'], fmt='-o', label='EA1: Max Fitness', color='black', capsize=3, linestyle='-', ecolor='black')
plt.errorbar(generations_EA2, data_EA2['max_fitness'], yerr=data_EA2['std_fitness'], fmt='--s', label='EA2: Max Fitness', color='darkgray', capsize=3, linestyle='--', ecolor='darkgray')

plt.errorbar(generations_EA1, data_EA1['avg_fitness'], yerr=data_EA1['std_avg_fitness'], fmt='-v', label='EA1: Avg Fitness', color='gray', capsize=3, linestyle='-', ecolor='gray')
plt.errorbar(generations_EA2, data_EA2['avg_fitness'], yerr=data_EA2['std_avg_fitness'], fmt='--x', label='EA2: Avg Fitness', color='lightgray', capsize=3, linestyle='--', ecolor='lightgray')

plt.plot(max_fitness_gen_EA1, max_fitness_value_EA1, 'ro', markersize=8, label=f'EA1: Max ({max_fitness_value_EA1:.2f})')
plt.plot(max_fitness_gen_EA2, max_fitness_value_EA2, 'bo', markersize=8, label=f'EA2: Max ({max_fitness_value_EA2:.2f})')

plt.text(max_fitness_gen_EA1, max_fitness_value_EA1 + 0.02, f'{max_fitness_value_EA1:.2f}', color='red', fontsize=10)
plt.text(max_fitness_gen_EA2, max_fitness_value_EA2 + 0.02, f'{max_fitness_value_EA2:.2f}', color='blue', fontsize=10)

plt.xlabel('Generation')
plt.ylabel('Fitness Metrics')
plt.title('Comparison of Fitness Metrics with Error Bars and Maximum Points for EA1 and EA2')

plt.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()