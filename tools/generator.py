import os
import pandas as pd
from sklearn.datasets import make_blobs


os.makedirs('data/input', exist_ok=True)

def generate_pow2_dataset(n_samples, n_features, centers, filename):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                      centers=centers, cluster_std=0.6, random_state=42)
    
    df = pd.DataFrame(X, columns=['x', 'y', 'z'])
    df['label'] = y
    
    full_path = os.path.join('data/input', filename)
    df.to_csv(full_path, index=False)
    print(f"Created: {full_path}")

pow2_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
for size in pow2_sizes:
    generate_pow2_dataset(size, 3, 5, f"test_{size}.csv")