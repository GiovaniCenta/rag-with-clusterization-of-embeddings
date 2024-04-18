import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters_umap(df, umap_component_prefix='umap component', cluster_column='cluster_label'):
    """
    Plota os clusters baseados nos componentes UMAP.

    Parâmetros:
    - df: DataFrame contendo os componentes UMAP e os rótulos de cluster.
    - umap_component_prefix: Prefixo usado nas colunas dos componentes UMAP.
    - cluster_column: Nome da coluna no DataFrame que contém os rótulos de cluster.
    """

    # Extrai os componentes UMAP
    umap_x = df[f'{umap_component_prefix} 1']
    umap_y = df[f'{umap_component_prefix} 2']
    
    # Configura o tamanho da figura
    plt.figure(figsize=(12, 8))
    
    # Utiliza Seaborn para plotar com uma paleta de cores variada baseada nos rótulos de cluster
    sns.scatterplot(x=umap_x, y=umap_y, hue=df[cluster_column], palette='Spectral', s=50, alpha=0.7, edgecolor='none')
    
    plt.title('Visualização dos Clusters com UMAP')
    plt.xlabel(f'{umap_component_prefix} 1')
    plt.ylabel(f'{umap_component_prefix} 2')
    
    # Exibe a legenda fora da área do gráfico
    plt.legend(title=cluster_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.show()


def create_umap(df, n_components=2, embedding_column_name='embedding'):
    

    # Converter lista de embeddings em um array numpy 2D
    X = np.array(df[embedding_column_name].tolist())

    # Normalizar os embeddings para terem norma unitária
    normalizer = Normalizer().fit(X)  # Ajustar normalizador para os embeddings
    X_normalized = normalizer.transform(X)  # Normalizar os embeddings

    # Realizar UMAP para reduzir os embeddings
    umap_reducer = UMAP(n_components=n_components, random_state=42)
    umapComponents = umap_reducer.fit_transform(X_normalized)  # Aplicar UMAP aos dados normalizados

    # Adicionar os componentes UMAP de volta ao dataframe
    for i in range(n_components):  # Ajustar com base no número de componentes de interesse
        df[f'umap component {i+1}'] = umapComponents[:, i]

    return umap_reducer, normalizer, n_components, df

def create_centroids_umap(df, n_components, component_prefix='umap component'):
    centroids = {}
    # Obter nomes únicos de documentos do DataFrame
    unique_documents = df['document_name'].unique()
    
    # Calcular o centróide para cada documento no espaço de maior dimensão reduzido pelo UMAP
    for document in unique_documents:
        document_df = df[df['document_name'] == document]
        # Criar uma lista para segurar os nomes das colunas para cada componente UMAP
        umap_columns = [f'{component_prefix} {i + 1}' for i in range(n_components)]
        # Garantir que document_df contém todos os componentes UMAP
        if all(col in document_df.columns for col in umap_columns):
            centroid = document_df[umap_columns].mean().values
            centroids[document] = centroid  # Armazenar o centróide
        else:
            print(f"Faltam componentes UMAP para {document}, certifique-se de que o UMAP foi aplicado com {n_components} componentes.")

    return centroids


def create_query_umap(query_embedding, umap_reducer, normalizer):
    # Normalizar o embedding de consulta usando o mesmo normalizador
    query_embedding_normalized = normalizer.transform(np.array(query_embedding).reshape(1, -1))

    # Aplicar UMAP ao embedding de consulta normalizado
    query_embedding_umap = umap_reducer.transform(query_embedding_normalized)

    return query_embedding_umap


from scipy.spatial.distance import cosine
import numpy as np

from scipy.spatial.distance import cosine

from scipy.spatial.distance import cosine
import numpy as np

def compare_similarity_umap(query_embedding_umap, chunks_df, umap_component_prefix='umap component'):
    similarities = {}

    # Assegura que o embedding da consulta está em formato 1-D
    query_embedding_umap = np.atleast_1d(query_embedding_umap.flatten())

    for index, row in chunks_df.iterrows():
        # Cria um vetor dos componentes UMAP para o trecho atual
        chunk_umap_embedding = np.atleast_1d(row[[f'{umap_component_prefix} {i + 1}' for i in range(query_embedding_umap.size)]]).astype(float)

        # Calcula a similaridade de cosseno
        cos_sim = 1 - cosine(query_embedding_umap, chunk_umap_embedding)

        # Armazena a similaridade
        similarities[index] = cos_sim

    return similarities

from scipy.spatial.distance import euclidean
import numpy as np

def compare_distance_umap(query_embedding_umap, chunks_df, umap_component_prefix='umap component'):
    distances = {}

    # Assegura que o embedding da consulta está em formato 1-D
    query_embedding_umap = np.atleast_1d(query_embedding_umap.flatten())

    for index, row in chunks_df.iterrows():
        # Cria um vetor dos componentes UMAP para o trecho atual
        chunk_umap_embedding = np.atleast_1d(row[[f'{umap_component_prefix} {i + 1}' for i in range(query_embedding_umap.size)]]).astype(float)

        # Calcula a distância euclidiana
        euclidean_dist = euclidean(query_embedding_umap, chunk_umap_embedding)

        # Armazena a distância
        distances[index] = euclidean_dist

    return distances





