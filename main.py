import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("customers.csv")

print("Dataset Preview:")
print(df.head())

# Features
X = df

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fix perplexity issue dynamically
n_samples = len(X_scaled)
perplexity_value = min(30, n_samples - 1)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Optional: Add clustering (important for portfolio)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
plt.title("Customer Segmentation using t-SNE")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
