import time
import json
import numpy as np

from cluster_methods.docname_methods import *
from utils.openai_utils import *
from utils.load_data import *
from find_chunk_methods.similarities_filter import *
from cluster_methods.hdbscan_methods import *
from cluster_methods.hclustering_methods import *
from cluster_methods.knn_methods import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np


class ClusteringCreation:
    """
    This class performs data clustering tasks.
    """

    def __init__(self, data_context, indexes, method='kmeans', n_components=127, show_clusters=False, llm_size=16200, chunk_size=1000, k_clusters=1):
        self.data_context = data_context
        self.indexes = indexes
        self.method = method
        self.n_components = n_components
        self.show_clusters = show_clusters
        self.llm_size = llm_size
        self.chunk_size = chunk_size
        self.k_clusters = k_clusters

    def configure_clustering(self, data):
        # Selecting clustering method
        if self.method == 'document':
            for i in range(data.shape[0]):
                data[i, self.indexes['cluster_label']] = data[i, self.indexes["document_name"]]
            centroids_parameters = self.create_centroids_document_name(data, self.indexes)

        elif self.method == 'hdbscan':
            centroids_parameters = self.create_centroids_hdbscan(data, self.indexes)

        elif self.method == 'hierarchical':
            centroids_parameters = self.create_centroids_hc(data, self.indexes)

        elif self.method.startswith('kmeans'):
            centroids_parameters = self.create_centroids_knn(data, self.indexes, self.method)

        centroids, centroid_document_map = centroids_parameters
        return centroids, centroid_document_map

    def create_pca(self, data, n_components=2, embedding_index=-2):
        # Assuming `data` is a structured array with an embedding at the last column by default
        embeddings = np.stack(data[:, embedding_index])
        # Normalize embeddings
        normalizer = Normalizer().fit(embeddings)
        X_normalized = normalizer.transform(embeddings)
        # Perform PCA
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(X_normalized)
        return pca, normalizer, principalComponents

    def create_centroids_document_name(self, data, indexes):
        return calculate_centroids_doc_name(data, indexes)

    def create_centroids_hdbscan(self, data, indexes):
        return calculate_hdbscan_centroids_with_pca(data, indexes)

    def create_centroids_hc(self, data, indexes):
        return calculate_centroids_hc(data, indexes)

    def create_centroids_knn(self, data, indexes, method, k_clusters=1):
        if '_' in method:
            # Extract the number of clusters directly from the method name, e.g., "kmeans_n_40"
            n_clusters = int(method.split('_')[-1])

            max_chunks_per_cluster = calculate_max_chunks_per_cluster(self.llm_size, self.chunk_size, n_clusters)

            # Calculate how many such clusters can be formed based on the number of rows in your data
            total_rows = len(data)
            max_clusters = calculate_max_clusters(total_rows, max_chunks_per_cluster)

            print(f"Max clusters: {max_clusters}")
            print(f"Max chunks per cluster: {max_chunks_per_cluster}")

        else:
            # Automatically determine the optimal number of clusters using the elbow method
            k_clusters = kmeans_elbow(data, indexes, print_elbow=False, max_clusters=200)
            max_clusters = k_clusters

        return create_centroids_knn(data, indexes, k_clusters=max_clusters)

    def create_clusters(self):
        data = self.data_context
        embedding_index = self.indexes['embedding']
        pca, normalizer, pca_components = self.create_pca(data, self.n_components, embedding_index)
        pca_components = pca_components.reshape(-1, 1) if pca_components.ndim == 1 else pca_components
        pca_components_object = np.empty((data.shape[0], 1), dtype=object)
        
        for i in range(data.shape[0]):
            pca_components_object[i, 0] = pca_components[i]

        for i in range(data.shape[0]):
            data[i, self.indexes['pca_components']] = pca_components[i]
        self.data_context=data
        
            
        # Normalizing PCA components
        centroids, centroid_document_map = self.configure_clustering(data)

        if self.show_clusters:
            plot_clusters(data, self.indexes)
            exit(8)

        return data, (pca, normalizer, pca_components, centroids, centroid_document_map)
