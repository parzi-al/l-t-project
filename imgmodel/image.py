import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# Load the dataset (MNIST-Like digits dataset from sklearn)
digits = load_digits()

# Separate data and labels
X = digits.data  # The image data (each image is 8x8, flattened to 64 features)
y = digits.target  # Labels for each image (digit: 0-9)

# Visualize some sample images from the dataset
def plot_sample_images(data, labels, n=10):
    plt.figure(figsize=(10, 5))
    for index, (image, label) in enumerate(zip(data[:n], labels[:n])):
        plt.subplot(2, n // 2, index + 1)
        plt.imshow(image.reshape(8, 8), cmap=plt.cm.gray)
        plt.title(f'Label: {label}')
        plt.axis('off')
    plt.savefig('../static/sample_images.png')
    plt.close()

plot_sample_images(X, y)

# 1. Dimensionality Reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 2. Visualizing the reduced data with PCA
def plot_pca(X_pca, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis", legend="full", s=60)
    plt.title('PCA: MNIST Digits in 2D')
    plt.savefig('../static/pca_plot.png')
    plt.close()

plot_pca(X_pca, y)

# 3. Dimensionality Reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 4. Visualizing the reduced data with t-SNE
def plot_tsne(X_tsne, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="coolwarm", legend="full", s=60)
    plt.title('t-SNE: MNIST Digits in 2D')
    plt.savefig('../static/tsne_plot.png')
    plt.close()

plot_tsne(X_tsne, y)

# 5. Analysis: Comparing PCA and t-SNE for MNIST Visualization
print(f'Explained variance by PCA components: {pca.explained_variance_ratio_}')
print(f'Total variance explained by 2 components: {np.sum(pca.explained_variance_ratio_):.4f}')

# 6. Save the PCA and t-SNE models
# Ensure the `models/` folder exists
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
os.makedirs(model_dir, exist_ok=True)

# Save the models with 'digits' as prefix
print("Saving models...")
joblib.dump(pca, os.path.join(model_dir, "digits_pca_model.pkl"))  # PCA Model
joblib.dump(tsne, os.path.join(model_dir, "digits_tsne_model.pkl"))  # t-SNE Model

print("Models saved successfully in the `models/` folder.")