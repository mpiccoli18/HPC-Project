import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

files = ["pack-report.txt", "pack-excl-report.txt", "scatter-report.txt", "scatter-excl-report.txt"]

for i, file in enumerate(files, start=1):
    
    data = pd.read_csv(file, sep=r'\s+|,', engine='python', header=None, names=['x', 'y'])

    plt.figure(figsize=(6, 4))
    plt.plot(data['x'], data['y'], marker='o', linestyle='-')
    plt.title("Plot {}: {}".format(i, file))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    plt.savefig(f"plot_{i}.png", dpi=300)
    
    plt.show()

print("All 4 plots generated successfully!")
