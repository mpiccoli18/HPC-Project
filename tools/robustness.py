import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt

true_data = pd.read_csv('./data/output/test_1000_clustered.csv')
y_true = true_data['label'].values

sigmas = [0.1, 0.5, 2.0, 5.0, 10.0]
numbers = [0, 1, 2, 3, 4]
nmi_results = []
ari_results = []

for n in numbers:
    pred_data = pd.read_csv(f'./data/accuracy/test_1000_{n}_clustered.csv')
    y_pred = pred_data['label'].values
    
    # Calculate Metrics
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    nmi_results.append(nmi)
    ari_results.append(ari)
    print(f"Sigma: {n} | NMI: {nmi:.4f} | ARI: {ari:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(sigmas, nmi_results, marker='o', label='NMI (Accuracy)')
plt.plot(sigmas, ari_results, marker='s', label='ARI (Stability)')
plt.xlabel('Sigma ($sigma$)')
plt.ylabel('Score')
plt.title('Robustness Analysis: Accuracy vs. Kernel Bandwidth')
plt.legend()
plt.grid(True)
plt.savefig('robustness_plot.png')