import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

def kmeans_elbow(data, indexes, print_elbow = False, max_clusters = 300):
    ## Prepare data for clustering: extract the PCA components
    X_pca = np.vstack(data[:, indexes["pca_components"]])

    ## Calculate inertia for a range of number of clusters
    inertia = []
    cluster_range = range(1, max_clusters + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_pca)
        inertia.append(kmeans.inertia_)

    ## Plot the elbow curve to find the optimal number of clusters
    if print_elbow:
        plt.figure(figsize=(10, 5))
        plt.plot(cluster_range, inertia, marker='o')
        plt.title('Elbow Method for Determining Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()

    # Use KneeLocator to find the elbow point
    kn = KneeLocator(cluster_range, inertia, curve='convex', direction='decreasing')
    print(f"Optimal number of clusters: {kn.knee}")
    return kn.knee




def apply_knn(data,indexes,k_clusters):
    # Preparar os dados para a clusterização: converter a coluna de componentes do PCA em um formato adequado
    X_pca = np.vstack(data[:, indexes["pca_components"]])
    knn = KMeans(n_clusters=k_clusters, random_state=0)
    knn.fit(X_pca)
    labels = knn.predict(X_pca)
    for i in range(data.shape[0]):
        data[i, indexes['cluster_label']] = labels[i]
        
    
    return knn, data

def create_centroids_knn(data, indexes,k_clusters):
    centroids = {}
    centroid_document_map = {}
    _,data = apply_knn(data,indexes,k_clusters)
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
        
    return centroids, centroid_document_map

def get_chunk_size(k,cluster_size, llm_size):
    chunk_size = llm_size/(k*cluster_size)
    return chunk_size

def split_chunks(data,indexes, chunk_size):
    new_chunks = []
    #aplit the columns of the data['chunk_text'] into chunks of size chunk_size, keep the other infos
    chunks = data[:,indexes['chunk_text']]
    for i in range(0, len(chunks), chunk_size):
        new_chunk = chunks[i:i + chunk_size]
        new_chunks.append(new_chunk)

    # modify the data to have the new chunks
def calculate_max_chunks_per_cluster(total_tokens_llm, chunk_size, num_clusters):
    """
    Calculate the maximum number of chunks per cluster given the token limit of the LLM.
    """
    total_chunks_possible = total_tokens_llm // chunk_size
    max_chunks_per_cluster = total_chunks_possible // num_clusters
    return max_chunks_per_cluster

def calculate_max_clusters(total_rows, chunks_per_cluster):
    """
    Calculate the maximum number of clusters that can be formed given the total rows and how many chunks fit in one cluster.
    """
    max_clusters = total_rows // chunks_per_cluster
    return max_clusters



def calculate_cluster_configuration(total_rows, tokens_per_row, total_tokens_llm, num_clusters):
    """
    Calculate the maximum number of clusters and chunks per cluster based on 
    the dataset size, chunk size, total token limit of the LLM, and desired number of clusters.

    Parameters:
        total_rows (int): Total number of rows in the dataset.
        tokens_per_row (int): Number of tokens in each row (chunk size).
        total_tokens_llm (int): The maximum token limit of the LLM.
        num_clusters (int): Desired number of clusters.

    Returns:
        tuple: (max_clusters, max_chunks_per_cluster) which are the maximum number of clusters possible
               and the maximum number of chunks that can be included in each cluster.
    """
    # Calculate the maximum number of rows per cluster based on token limits
    max_rows_per_cluster = total_tokens_llm // tokens_per_row // num_clusters
    if max_rows_per_cluster == 0:
        raise ValueError("The token limit per cluster is too small to include even one row.")

    # Calculate the total number of clusters that can be formed
    max_clusters = total_rows // max_rows_per_cluster

    # Calculate the maximum number of chunks per cluster
    total_chunks_possible = total_tokens_llm // tokens_per_row
    max_chunks_per_cluster = total_chunks_possible // num_clusters

    print(f"Each cluster can contain up to {max_rows_per_cluster} chunks.")
    print(f"Maximum number of clusters possible: {max_clusters}")
    print(f"Maximum number of chunks per cluster: {max_chunks_per_cluster}")

    return max_clusters, max_chunks_per_cluster


# Example usage:
total_tokens_llm = 16000  # Total token limit for the LLM
chunk_size = 1000        # Number of tokens per chunk
num_clusters = 4         # Desired number of clusters

max_chunks_per_cluster = calculate_max_chunks_per_cluster(total_tokens_llm, chunk_size, num_clusters)
