import sys
sys.path.append("..") # Adds higher directory to python modules path.
import hdbscan
from cluster_methods.hdbscan_methods import *
from cluster_methods.umap_methods import *
from cluster_methods.cluster_functions import *
from utils.query_utils import *
from utils.openai_utils import *
from find_chunk_methods.search_functions import *
from find_chunk_methods.find_chunks import *
from utils.plot_utils import *

from scipy.spatial.distance import euclidean
import numpy as np

def apply_hdbscan(df, min_cluster_size=5, min_samples=None, embedding_column_name='pca_components'):
    # Preparar os dados para a clusterização: converter a coluna de componentes do PCA em um formato adequado
    X_pca = np.vstack(df[embedding_column_name].values)
    
    # Aplicar HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X_pca)
    
    # Adicionar os rótulos dos clusters ao DataFrame
    df['cluster_label'] = labels
    return clusterer, df

def calculate_hdbscan_centroids_with_pca(df, embedding_column_name='pca_components'):
    # Assumindo que a função apply_hdbscan já foi aplicada e as etiquetas de clusters estão no df
    centroids = {}
    _,df = apply_hdbscan(df)
    unique_clusters = df['cluster_label'].unique()
    
    # Processa clusters normais, excluindo ruído
    for cluster in unique_clusters:
        if cluster == -1:  # Ignora ruído nesta etapa
            continue
        cluster_df = df[df['cluster_label'] == cluster]
        # Extrai os vetores PCA para os pontos no cluster atual
        pca_vectors = np.vstack(cluster_df[embedding_column_name].values)
        centroid = pca_vectors.mean(axis=0)
        centroids[cluster] = centroid
    
    # Trata dos pontos de ruído, agrupando-os por 'document_name'
    noise_centroids = handle_noise_by_document(df, embedding_column_name)
    
    # Combina centróides dos clusters normais com os dos "clusters de ruído"
    centroids.update(noise_centroids)
    
    return centroids



def handle_noise_by_document(df, embedding_column_name='pca_components'):
    centroids = {}
    noise_df = df[df['cluster_label'] == -1]
    document_names = noise_df['document_name'].unique()
    
    for document in document_names:
        doc_df = noise_df[noise_df['document_name'] == document]
        pca_vectors = np.vstack(doc_df[embedding_column_name].values)
        centroid = pca_vectors.mean(axis=0)
        # Cria um identificador de cluster único para estes novos "clusters" baseado no nome do documento
        cluster_id = f'noise_{document}'
        centroids[cluster_id] = centroid
    
    return centroids

