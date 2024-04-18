
import hdbscan
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_methods.hdbscan_methods import *
from cluster_methods.umap_methods import *
from cluster_methods.cluster_functions import *
from utils.query_utils import *
from utils.openai_utils import *
from find_chunk_methods.search_functions import *
from find_chunk_methods.find_chunks import *
from utils.plot_utils import *
import numpy as np



def find_chunks_in_multiple_documents_euclidean_kdtree(centroids, query_pca, query_embedding, df, n_components=128, top_contexts=5, distance_threshold=0.5, print_distances=True):
    # Converter dicionário de centróides para array para construir KDTree
    centroid_list = list(centroids.values())
    centroid_keys = list(centroids.keys())
    tree = KDTree(centroid_list)
    
    # Encontrar os índices dos centróides mais próximos e suas distâncias
    distances, indices = tree.query(query_pca.reshape(1, -1), k=len(centroid_keys))
    
    # Mapear índices para as chaves dos centróides e criar um dicionário de distâncias
    distances_dict = {centroid_keys[i]: dist for i, dist in zip(indices[0], distances[0])}
    
    # Imprimir distâncias se necessário
    if print_distances:
        call_print_distances(distances_dict)
        

    # Aplicar o threshold de distância para determinar os centróides mais próximos válidos
    #closest_books = [centroid_keys[i] for i, dist in zip(indices[0], distances[0]) if dist <= distance_threshold]
    closest_books = filter_closest_books(query_pca, centroids, distance_threshold)
    


    
    
    # Buscar os melhores chunks dentro dos documentos dos centróides mais próximos
    best_chunks_info, best_similarities = find_best_chunks_kdtree(top_contexts, query_pca, closest_books, df, embedding_column_name='pca_components')
    
    return best_chunks_info, best_similarities

def call_print_distances(distances):
    # Ordenar e imprimir as distâncias de forma clara
    for book, distance in sorted(distances.items(), key=lambda item: item[1]):
        print(f"Distance from query to centroid of '{book}': {distance}")





#call main
if __name__ == "__main__":
    # Load the data
    df = pd.read_pickle('hp_dataset_1500_summarized.pkl')
    df['embeddings_summary'] = df['embeddings_summary'].apply(lambda x: x.data[0].embedding)
    pca, normalizer,n_components,df  = create_pca(df,n_components=12,embedding_column_name='embeddings_summary')
    centroids = create_centroids(df,n_components=n_components)
    #df,fig,centroids = plot_clusters_2_components(df,plot=False)
    distance_threshold = 0.4
    query_text = "Relação entre Sirius e Dumbledore"
    query_embedding = create_query_embedding(query_text)
    query_pca = create_query_pca(pca,normalizer,query_embedding)
    best_chunks_info, best_similarities = find_chunks_in_multiple_documents_euclidean_kdtree(centroids, query_pca, query_embedding, df,distance_threshold=distance_threshold)
    show_findings(best_chunks_info, best_similarities)
    #plot_query_2_components(query_pca,fig,query_text,plot=True)