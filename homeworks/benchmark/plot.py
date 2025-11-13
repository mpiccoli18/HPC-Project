import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

files = ["pack-report.txt", "pack-excl-report.txt", "scatter-report.txt", "scatter-excl-report.txt"]

for i, file in enumerate(files, start=1):
    
    data = pd.read_csv(file, sep=r'\s+|,', engine='python', header=None, skiprows=1, usecols=[0, 1], names=['x', 'y'])
    
    
    plt.figure(figsize=(7, 5))
    plt.plot(data['x'], data['y'], label= 'Report', marker='o', linestyle='-')
    
    plt.plot(data['x'], (data['x'] * 2) / data['y'],  label= 'Bandwidth', color='red', marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Grafico {}: {}".format(i, file))
    plt.xlabel("X (log scale)")
    plt.ylabel("Y (log scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f"plot_{i}.png", dpi=300)
    
    plt.show()

print("Generati correttamente 4 grafici!")
