import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from preprocessing import RANDOM_STATE, run as run_preprocessing

def preprocess_special():
    # Import preprocessed data from the shared preprocessing pipeline.
    X_train, y_train, X_val, X_test, y_val, y_test, preprocessor = run_preprocessing()
    return X_train, X_test, y_train, y_test

def run():

    X_train, X_test, y_train, y_test = preprocess_special()

    # Apply PCA and analyze explained variance:

    # how many components are neede for enough information
    # line at 99% and see where curve crosses it
    # feature number at linetells us how many features are really relevant (redundancy in original features)
    pca = PCA(random_state=RANDOM_STATE).fit(X_train)
    n_components = (pca.explained_variance_ratio_.cumsum() < 0.99).sum()
    print(f"Components needed for 99% variance: {n_components}")

    # Cumulative explained variance plot
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(0.99, color='red', linestyle='--', label='99% threshold')
    plt.axvline(n_components, color='orange', linestyle='--', label=f'n={n_components}')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA - Explained Variance')
    plt.legend()
    plt.savefig('phases/results/USA/pca_explained_variance.png')
    plt.show()

    # refit with the chosen number of components(lab11)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Visualise the first two principal components coloured by class
    # Plot shows that class differences are subtle and live in higher dimensions
    plt.figure()
    for cls in np.unique(y_train):
        mask = y_train == cls
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], s=15, alpha=0.6, label=str(cls))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA - First 2 components (train set)')
    plt.legend(title='Class')
    plt.savefig('phases/results/USA/pca_scatter.png')
    plt.show()

    # K-Means: Elbow + Silhouette 
    # Inertia = how tightly are the points in each cluster packed together (sum of squared distances from each point to its cluster center)
    # inertia always decreases as you add more clusters (with k = n_samples, inertia = 0)
    inertias = []
    silhouettes_km = []
    k_range = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_train_pca)
        inertias.append(km.inertia_)
        silhouettes_km.append(silhouette_score(X_train_pca, labels))

    # Elbow plot
    # elbow: the point where adding another cluster stops giving a big improvement.
    plt.figure()
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('K-Means - Elbow Method')
    plt.savefig('phases/results/USA/kmeans_elbow.png')
    plt.show()

    # Silhouette plot
    # how well each point fits in its assigned cluster compared to the nearest neighboring cluster
    # ranges from -1 to +1: 
    # close to +1 means points are well inside their cluster and far from others
    # close to 0 means points sit on the boundary between clusters
    # negative means points are probably in the wrong cluster
    # You plot the average silhouette for each k and pick the k with the highest score -> very accurate
    # Together with the elbow plot, you can justify your choice of k
    plt.figure()
    plt.plot(list(k_range), silhouettes_km, marker='o', color='green')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('K-Means - Silhouette Score')
    plt.savefig('phases/results/USA/kmeans_silhouette.png')
    plt.show()

    best_k_km = list(k_range)[silhouettes_km.index(max(silhouettes_km))]
    print(f"Best k for K-Means (silhouette): {best_k_km}")

    # GMM: Elbow + Silhouette
    neg_log_likelihoods = []
    silhouettes_gmm = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
        gmm.fit(X_train_pca)
        labels = gmm.predict(X_train_pca)
        neg_log_likelihoods.append(-gmm.score(X_train_pca) * len(X_train_pca))
        silhouettes_gmm.append(silhouette_score(X_train_pca, labels))

    # Elbow plot for GMM (negative log-likelihood)
    plt.figure()
    plt.plot(list(k_range), neg_log_likelihoods, marker='o')
    plt.xlabel('k')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('GMM - Elbow Method')
    plt.savefig('gmm_elbow.png')
    plt.show()

    # Silhouette plot
    plt.figure()
    plt.plot(list(k_range), silhouettes_gmm, marker='o', color='green')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('GMM - Silhouette Score')
    plt.savefig('phases/results/USA/gmm_silhouette.png')
    plt.show()

    best_k_gmm = list(k_range)[silhouettes_gmm.index(max(silhouettes_gmm))]
    print(f"Best k for GMM (silhouette): {best_k_gmm}")

    # Fit final models and compare clusters vs true labels
    km_final = KMeans(n_clusters=best_k_km, random_state=RANDOM_STATE, n_init=10)
    km_final.fit(X_train_pca)
    km_labels = km_final.predict(X_train_pca)

    gmm_final = GaussianMixture(n_components=best_k_gmm, random_state=RANDOM_STATE, n_init=5)
    gmm_final.fit(X_train_pca)
    gmm_labels = gmm_final.predict(X_train_pca)

    # Three side-by-side scatter plots in PCA space, all showing the same data points but colored differently: 
    # the left panel uses the true class labels
    # the middle uses K-Means cluster assignments
    # the right uses GMM cluster assignments
    # Side-by-side scatter: true labels vs cluster assignments

    # If cluster coloring looks similar to the true-label coloring, it means the natural structure of the data (without ever looking at labels) already reflects the target variable -> supervised models 
    # GMM plot matches perfectly but classes are switched up

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for cls in np.unique(y_train):
        mask = y_train == cls
        axes[0].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], s=10, alpha=0.5, label=str(cls))
    axes[0].set_title('True Labels')
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].legend(title='Class', fontsize=7)

    for cls in np.unique(km_labels):
        mask = km_labels == cls
        axes[1].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], s=10, alpha=0.5, label=str(cls))
    axes[1].set_title(f'K-Means (k={best_k_km})')
    axes[1].set_xlabel('PC 1')
    axes[1].legend(title='Cluster', fontsize=7)

    for cls in np.unique(gmm_labels):
        mask = gmm_labels == cls
        axes[2].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], s=10, alpha=0.5, label=str(cls))
    axes[2].set_title(f'GMM (k={best_k_gmm})')
    axes[2].set_xlabel('PC 1')
    axes[2].legend(title='Cluster', fontsize=7)

    plt.suptitle('Cluster Assignments vs True Labels')
    plt.tight_layout()
    plt.savefig('phases/results/USA/cluster_vs_target.png')
    plt.show()

    # Summary table
    results = pd.DataFrame({
        'Model':      ['K-Means', 'GMM'],
        'Best k':     [best_k_km, best_k_gmm],
        'Silhouette': [round(max(silhouettes_km), 4), round(max(silhouettes_gmm), 4)],
    })
    print(results)
    return results


if __name__ == "__main__":
    run()