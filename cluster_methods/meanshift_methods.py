from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import euclidean
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_methods.hdbscan_methods import *
from cluster_methods.docname_methods import *
from utils.query_utils import *
from utils.openai_utils import *

from utils.plot_utils import *
# Assumindo a existência das funções necessárias de pré-processamento e comparação de distâncias em um arquivo auxiliar.

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import euclidean
# Assume a existência das funções necessárias de pré-processamento e comparação de distâncias em um arquivo auxiliar.

def apply_meanshift(df, umap_components_prefix='umap component', quantile=0.2, n_samples=500):
    # Extrair apenas os componentes UMAP
    umap_columns = [f'{umap_components_prefix} {i+1}' for i in range(len(df.filter(regex=umap_components_prefix).columns))]
    X_umap = df[umap_columns].values

    # Padronizar os dados antes de aplicar o MeanShift
    X_umap_standardized = StandardScaler().fit_transform(X_umap)

    # Estimar a largura de banda para o MeanShift
    bandwidth = estimate_bandwidth(X_umap_standardized, quantile=quantile, n_samples=n_samples)

    # Aplicar o MeanShift
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = clusterer.fit_predict(X_umap_standardized)

    # Adicionar os rótulos dos clusters ao DataFrame
    df['cluster_label'] = labels
    return clusterer, df

def calculate_meanshift_centroids(df, n_components,umap_components_prefix='umap component'):
    _, df = apply_meanshift(df)  # Aplica MeanShift e atualiza df
    centroids = {}
    unique_clusters = df['cluster_label'].unique()

    for cluster in unique_clusters:
        cluster_df = df[df['cluster_label'] == cluster]
        umap_columns = [f'{umap_components_prefix} {i+1}' for i in range(len(cluster_df.filter(regex=umap_components_prefix).columns))]
        centroid = cluster_df[umap_columns].mean().values
        centroids[cluster] = centroid

    return centroids

def find_closest_clusters_meanshift(query_embedding, centroids, distance_threshold=0.1):
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()

    distances = {cluster: euclidean(query_embedding, centroid) for cluster, centroid in centroids.items()}
    sorted_distances = sorted(distances.items(), key=lambda item: item[1])
    closest_distance = sorted_distances[0][1] if sorted_distances else float('inf')
    clusters_within_threshold = [cluster for cluster, distance in sorted_distances if distance <= closest_distance + distance_threshold]

    return clusters_within_threshold



def find_chunks_in_clusters_meanshift(centroids, query_embedding, df, distance_threshold=0.1, embedding_column_name='embedding'):
    # Calcular distâncias Euclidianas do embedding de consulta a cada centróide
    query_embedding = query_embedding.flatten()
    distances = {}
    for cluster, centroid in centroids.items():
        # Certifique-se de que cada centróide é um vetor 1-D
        centroid = np.array(centroid).flatten()
        
        distance = euclidean(query_embedding, centroid)
        distances[cluster] = distance
    

    
  
