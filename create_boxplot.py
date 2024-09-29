import numpy as np
import matplotlib.pyplot as plt


blendx2 = [98, 80, 70, 70, 98]
blendx7 = [-40, -40, 91, -20, 91]
blendx8 = [-20, 91, 91, -20, 91]

twopointx2 = [72, 92, 52, 92, 72]
twopointx7 = [-30, 94.6, 94.6, 94.6, -80]
twopointx8 = [88.6, 61.6, 61.6, 88.6, 58.6]


data = [blendx2, twopointx2, blendx7, twopointx7, blendx8, twopointx8]

fig, ax = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.size': 10})
ax.boxplot(data, showmeans=True)

# Add labels and legend (title is added in latex)
# plt.xlabel('EA specialist instances')
plt.ylabel('Individual gain')

# Edit x-limits and ticks
plt.xticks([1, 1.5, 2, 3, 3.5, 4, 5, 5.5, 6],
           ['BlendX', '\n\n| - - - - - - - - Enemy 2 - - - - - - - - |', '2-PointX',
            'BlendX', '\n\n| - - - - - - - - Enemy 7 - - - - - - - - |', '2-PointX',
            'BlendX', '\n\n| - - - - - - - - Enemy 8 - - - - - - - - |', '2-PointX'])
ax.tick_params(axis='x', which='both',length=0)

plt.tight_layout()
plt.show()
