# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import h5py

# Load the dataset (MNIST-like digits dataset from sklearn)
digits = load_digits()

# Separate data and labels
X = digits.data  # The image data (each image is 8x8, flattened to 64 features)
y = digits.target  # Labels for each image (digits: 0-9)

# 1. Visualize some sample images from the dataset
def plot_sample_images(data, labels, n=10):
    plt.figure(figsize=(10, 5))
    for i in range(1, n + 1):
        plt.subplot(1, n, i)
        plt.imshow(data[i - 1].reshape(8, 8), cmap=plt.cm.gray)
        plt.title(f"Label: {labels[i - 1]}")
        plt.axis("off")
    plt.show()

plot_sample_images(X, y)

# 2. Dimensionality Reduction using PCA
# Reduce data from 64 dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# a. Visualizing the reduced data with PCA
def plot_pca(X_pca, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis", legend="full", s=40)
    plt.title("PCA: MNIST Digits in 2D")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

plot_pca(X_pca, y)

# 3. Dimensionality Reduction using t-SNE
# t-SNE is often better at preserving local structure in high-dimensional data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# b. Visualizing the reduced data with t-SNE
def plot_tsne(X_tsne, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="coolwarm", legend="full", s=40)
    plt.title("t-SNE: MNIST Digits in 2D")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

plot_tsne(X_tsne, y)

# 4. Comparing PCA and t-SNE for MNIST Visualization
# Calculate total variance for PCA to understand how much information is retained
explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"Variance explained by PCA components: {explained_variance:.2f}%")

# 5. Save PCA and t-SNE results to an HDF5 file
with h5py.File('image.h5', 'w') as h5f:
    h5f.create_dataset('PCA', data=X_pca)
    h5f.create_dataset('t-SNE', data=X_tsne)
    h5f.create_dataset('Labels', data=y)

print("\nPCA and t-SNE results have been saved to 'image.h5'.")
