from utils.openai_utils import *
from utils.load_data import *
from find_chunk_methods.similarities_filter import *
from cluster_methods.hdbscan_methods import *

from cluster_methods.hclustering_methods import *
from cluster_methods.knn_methods import *
import time
import json

import numpy as np
from scipy.spatial.kdtree import KDTree


class ClosestClustersToQuery:
    """
    This class finds closest clusters to a given query.
    """

    def __init__(self, data, indexes, query_text, n_components, clusters_parameters,inside_thr=0.08, distance_threshold=0.1, show_clusters_distances = False):
        """
        Initializes the class with parameters for finding closest clusters.

        Args:
          data: The data context.
          indexes: Dictionary containing indexes for data elements.
          query_text: The query text to find closest clusters for.
          n_components: Number of components for PCA.
          inside_thr: Threshold for considering a document to be inside a cluster (default: 0.08).
          distance_threshold: Threshold for filtering closest documents (default: 0.1).
          k: Number of closest clusters to return (default: 1).
          show_results: Whether to print the results (default: False).
        """

        self.data = data
        self.indexes = indexes
        self.query_text = query_text
        self.n_components = n_components
        self.inside_thr = inside_thr
        self.distance_threshold = distance_threshold
        self.show_clusters_distances = show_clusters_distances

        # Pre-calculated parameters for efficiency (assumed to be available externally)
        self.pca, self.normalizer, self.pca_components, self.centroids, self.centroid_document_map = clusters_parameters

    def find_closest_clusters(self):
        """
        Finds closest clusters to the query text.

        Returns:
        A list of document names closest to the query.
        """

        if self.pca is None or self.normalizer is None or self.pca_components is None or self.centroids is None or self.centroid_document_map is None:
            raise ValueError("Pre-calculated parameters (pca, normalizer, pca_components, centroids, centroid_document_map) are not set.")

        query_pca = self.pca_query(self.query_text, self.pca, self.normalizer)

        centroids = np.array([info[1] for info in self.centroid_document_map.values()])  # Extract numerical PCA components
        document_names = [info[0] for info in self.centroid_document_map.values()]  # Extract corresponding document names

        tree = KDTree(centroids)
        distances, indices = tree.query(query_pca.reshape(1, -1), k=len(centroids))
        distances_dict = {document_names[idx]: dist for idx, dist in zip(indices[0], distances[0])}

        # Sort the distances to find the closest centroid
        sorted_distances = sorted(distances_dict.items(), key=lambda item: item[1])

        if self.distance_threshold is None:
            print("Calculating median distance threshold...")
            self.distance_threshold = np.median(distances)

        closest_distance = sorted_distances[0][1] if sorted_distances else float('inf')

        # Filter for document names within the threshold distance of the closest centroid
        closest_documents = [document for document, distance in sorted_distances if distance <= closest_distance + self.distance_threshold]

        if self.show_clusters_distances:
            #print cluster distance with index number
            print("Sorted distances and document names within threshold:")
            for document, distance in sorted_distances:
                if distance <= closest_distance + self.distance_threshold:
                    print(f"Distance to centroid {document}: {distance}")
            

        return closest_documents, query_pca

    def pca_query(self, query_text, pca, normalizer):
        """
        Transforms the query text using PCA.

        Args:
            query_text: The query text.
            pca: The PCA object.
            normalizer: The normalizer object.

        Returns:
            The query text transformed using PCA.
        """

        query_embedding = create_query_embedding_ai_gateway(query_text)
        query_pca = create_query_pca(pca, normalizer, query_embedding)
        return query_pca
