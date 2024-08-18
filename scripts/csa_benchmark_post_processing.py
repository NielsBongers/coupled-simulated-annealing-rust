from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results/ackley_benchmark.csv", sep=",")

grouped = df.groupby('number_threads')['performance']

mean_performance = grouped.mean()
variance_performance = grouped.var()

plt.figure(figsize=(10, 6))
plt.plot(mean_performance.index, mean_performance, label='Mean', color='b')
plt.fill_between(mean_performance.index, 
                 mean_performance - np.sqrt(variance_performance), 
                 mean_performance + np.sqrt(variance_performance), 
                 color='b', alpha=0.2, label='Variance')
plt.xlabel('Threads')
plt.ylabel('Performance')
plt.title('Coupled simulated annealing performance')
plt.legend()
plt.xticks(ticks=np.arange(1, len(grouped.unique()) + 1, 1))  # Setting x-axis ticks only at integer values
plt.semilogy()
plt.grid(True)

Path("figures").mkdir(exist_ok=True, parents=True)
plt.savefig(f"figures/{datetime.now().strftime('%d%m%Y - ')}Coupled simulated annealing - performance comparison.png", dpi=300)

plt.show()