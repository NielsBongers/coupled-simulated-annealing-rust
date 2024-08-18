from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results/ackley_benchmark.csv", sep=",")

grouped = df.groupby('number_threads')['performance']

mean_performance = grouped.mean() + 1.0
best_performance = grouped.min() + 1.0
worst_performance = grouped.max() + 1.0

plt.figure(figsize=(10, 6))
plt.plot(mean_performance.index, mean_performance, label='Mean', color='b')
plt.plot(mean_performance.index, best_performance, label='Best', color='g')
plt.plot(mean_performance.index, worst_performance, label='Worst', color='r')

plt.fill_between(mean_performance.index, 
                 best_performance, 
                 worst_performance, 
                 color='b', alpha=0.2)

plt.xlabel('Threads')
plt.ylabel('Performance')
plt.title('Coupled simulated annealing, 40.000 evaluations overall on Ackley')
plt.legend()
plt.xticks(ticks=np.arange(1, len(grouped.unique()) + 1, 1))  # Setting x-axis ticks only at integer values
plt.semilogy()
plt.grid(True)

Path("figures").mkdir(exist_ok=True, parents=True)
plt.savefig(f"figures/{datetime.now().strftime('%d%m%Y - ')}Coupled simulated annealing - performance comparison - 40.000 total.png", dpi=300)

plt.show()