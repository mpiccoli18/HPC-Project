import pandas as pd
import matplotlib.pyplot as plt
import sys

files = ['pack-report.txt', 'pack-excl-report.txt', 'scatter-report.txt', 'scatter-excl-report.txt']
labels = ['pack', 'pack:excl', 'scatter', 'scatter:excl']


def plot_average_time():
    plt.figure(figsize=(10, 6))
    
    for file, label in zip(files, labels):
        df = pd.read_csv(file, delim_whitespace=True)
        plt.plot(df['bytes'], df['average_t'], marker='o', label=label)

    plt.xscale('log', base=2)
    plt.xlabel('Bytes', fontsize=12)
    plt.ylabel('Average time (s)', fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    plot_average_time()
