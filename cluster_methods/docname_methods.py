import pandas as pd
import numpy as np


def calculate_centroids_doc_name(data, indexes):
    centroids = []
    centroid_document_map = {}

    
    for i, document_name in enumerate(np.unique(data[:, indexes['document_name']])):
        document_data = data[data[:, indexes['document_name']] == document_name]
        centroid = np.mean(document_data[:, indexes['pca_components']], axis=0)
        centroids.append(centroid)
        centroid_document_map[i] = document_name,centroid  # Map index to document name

    return np.array(centroids), centroid_document_map







