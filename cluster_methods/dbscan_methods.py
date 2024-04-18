from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import euclidean

import hdbscan
from search_functions import *
from umap_methods import *
# Assumindo que compare_distance_umap já esteja definida em search_functions
# from search_functions import compare_distance_umap

def apply_dbscan(df, eps=0.5, min_samples=2, umap_components_prefix='umap component'):
    # Extrair apenas os componentes UMAP
    umap_columns = [f'{umap_components_prefix} {i+1}' for i in range(len(df.filter(regex=umap_components_prefix).columns))]
    X_umap = df[umap_columns].values

    # Padronizar os dados antes de aplicar DBSCAN
    X_umap_standardized = StandardScaler().fit_transform(X_umap)

    # Aplicar DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(X_umap_standardized)

    # Adicionar os rótulos dos clusters ao DataFrame
    df['cluster_label'] = labels
    return clusterer, df

def calculate_dbscan_centroids(df, n_components, umap_components_prefix='umap component'):
    _, df = apply_dbscan(df)  # Aplica DBSCAN e atualiza df
    centroids = {}
    unique_clusters = df['cluster_label'].unique()

    # Processa primeiro os clusters normais (excluindo ruído)
    for cluster in unique_clusters:
        if cluster == -1:  # Pula os pontos considerados ruído nesta etapa
            continue
        cluster_df = df[df['cluster_label'] == cluster]
        umap_columns = [f'{umap_components_prefix} {i+1}' for i in range(len(cluster_df.filter(regex=umap_components_prefix).columns))]
        centroid = cluster_df[umap_columns].mean().values
        centroids[cluster] = centroid

    # Agora, trata dos pontos de ruído usando a função específica
    noise_centroids = handle_noise_by_document(df, umap_components_prefix, n_components)
    
    # Combina os centróides dos clusters normais com os centróides dos "clusters de ruído"
    centroids.update(noise_centroids)

    return centroids


def handle_noise_by_document(df, umap_components_prefix='umap component', n_components=18):
    
    centroids = {}
    noise_df = df[df['cluster_label'] == -1]
    document_names = noise_df['document_name'].unique()
    
    for document in document_names:
        doc_df = noise_df[noise_df['document_name'] == document]
        umap_columns = [f'{umap_components_prefix} {i+1}' for i in range(n_components)]
        centroid = doc_df[umap_columns].mean().values
        # Cria um identificador de cluster único para estes novos "clusters" baseado no nome do documento
        cluster_id = f'noise_{document}'
        centroids[cluster_id] = centroid
    
    return centroids


def find_closest_clusters_dbscan(query_embedding, centroids, distance_threshold=0.1):
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()

    distances = {cluster: euclidean(query_embedding, centroid) for cluster, centroid in centroids.items() if cluster != -1}
    sorted_distances = sorted(distances.items(), key=lambda item: item[1])
    closest_distance = sorted_distances[0][1] if sorted_distances else float('inf')
    clusters_within_threshold = [cluster for cluster, distance in sorted_distances if distance <= closest_distance + distance_threshold]

    return clusters_within_threshold

def find_best_chunks_dbscan(query_embedding, closest_clusters, df, embedding_column_name='embedding'):
    all_distances = []

    for cluster in closest_clusters:
        cluster_chunks = df[df['cluster_label'] == cluster]
        distances = compare_distance_umap(query_embedding, cluster_chunks, umap_component_prefix=embedding_column_name)
        
        for idx, distance in distances.items():
            if idx:
                all_distances.append((distance, idx))

    all_distances.sort(key=lambda x: x[0])
    top_distances = all_distances[:5]
    best_chunks_info = [df.loc[idx] for _, idx in top_distances]

    return best_chunks_info, [dist for dist, _ in top_distances]

from scipy.spatial.distance import euclidean

def find_chunks_in_clusters_dbscan(centroids, query_embedding, df, distance_threshold=0.1, embedding_column_name='embedding'):
    # Calcular distâncias Euclidianas do embedding de consulta a cada centróide
    query_embedding = query_embedding.flatten()
    distances = {}
    for cluster, centroid in centroids.items():
        # Certifique-se de que cada centróide é um vetor 1-D
        centroid = np.array(centroid).flatten()
        
        distance = euclidean(query_embedding, centroid)
        distances[cluster] = distance
    

    for cluster, distance in sorted(distances.items(), key=lambda item: item[1]):
        print(f"Distance from query to centroid of cluster '{cluster}': {distance}")

    # Identificar os clusters mais próximos com base na distância Euclidiana
    closest_clusters = find_closest_clusters_dbscan(query_embedding, centroids, distance_threshold=distance_threshold)
    print("Searching for the best chunks in the following clusters:", closest_clusters)
  
    ## ok
    # Adaptar a chamada à função para encontrar os melhores trechos dentro dos clusters identificados
    best_chunks_info, best_similarities = find_best_chunks_dbscan(query_embedding, closest_clusters, df, embedding_column_name)

    return best_chunks_info, best_similarities