import sys
sys.path.append("..") # Adds higher directory to python modules path.
import hdbscan
from cluster_methods.hdbscan_methods import *
#from cluster_methods.umap_methods import *
from utils.query_utils import *
from utils.openai_utils import *
from utils.plot_utils import *

from scipy.spatial.distance import euclidean
import numpy as np

def apply_hdbscan(data,indexes, min_cluster_size=3, min_samples=None):
    # Preparar os dados para a clusterização: converter a coluna de componentes do PCA em um formato adequado
    X_pca = np.vstack(data[:, indexes["pca_components"]])
    
    # Aplicar HDBSCA
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X_pca)
    
    # Adicionar os rótulos dos clusters ao DataFrame
    for i in range(data.shape[0]):
        data[i, indexes['cluster_label']] = labels[i]
        
    
    return clusterer, data

def calculate_hdbscan_centroids_with_pca(data,indexes):
    # Assumindo que a função apply_hdbscan já foi aplicada e as etiquetas de clusters estão no df
    centroids = {}
    centroid_document_map = {}
    _,data = apply_hdbscan(data,indexes)
    cluster_labels = data[:, indexes['cluster_label']]
    unique_clusters = np.unique(cluster_labels)
    
    
    for cluster in unique_clusters:
        if cluster == -1:  # Ignore noise
            continue
        mask = cluster_labels == cluster
        cluster_data = data[mask]
        
        # Extract PCA vectors for points in the current cluster
        pca_vectors = np.vstack(cluster_data[:, indexes['pca_components']])
        centroid = pca_vectors.mean(axis=0)
        centroids[cluster] = centroid
        centroid_document_map[cluster] = cluster,centroid 
    
    # Optionally handle noise based on document name or other criteria
    noise_centroids_dict = handle_noise_by_document(data, indexes)
    centroid_document_map.update(noise_centroids_dict)
    
    
    return centroids, centroid_document_map


def handle_noise_by_document(data,indexes):
    centroids = {}
    
    noise_mask = data[:, indexes['cluster_label']] == -1
    noise_data = data[noise_mask]
    document_names = np.unique(noise_data[:,indexes['document_name']])
    centroid_document_map = {}
    for document in document_names:
        # Crie uma máscara booleana onde a condição é verdadeira
        mask = noise_data[:, indexes['document_name']] == document
        # Use a máscara para filtrar as linhas
        doc_data = noise_data[mask]
        if doc_data.size > 0:
            pca_vectors = np.vstack(doc_data[:, indexes['pca_components']])
            centroid = pca_vectors.mean(axis=0)
            cluster_id = f'noise_{document}'
            centroids[cluster_id] = centroid
            centroid_document_map[cluster_id] = cluster_id,centroid 

    return centroid_document_map


