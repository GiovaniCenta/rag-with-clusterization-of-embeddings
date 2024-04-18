
import os
import pandas as pd
import openai
from dotenv import load_dotenv
import PyPDF2  # Import PyPDF2
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_methods.hdbscan_methods import *
from cluster_methods.cluster_functions import *
from utils.query_utils import *
from utils.openai_utils import *
from find_chunk_methods.search_functions import *
from find_chunk_methods.find_chunks import *
from utils.plot_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances    
import datetime
import time
#import warnings
# Suprimir apenas PerformanceWarning
#warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def configure_clustering(df, method='document', n_components=127, embedding_column_name='embedding'):
    pca, normalizer, n_components, df = create_pca(df, n_components, embedding_column_name)
    if method == 'document':
        # Clusterização por nome de documento
        df['cluster_label'] = df['document_name']
        centroids = create_centroids(df,embedding_column_name='pca_components')
    elif method == 'hdbscan':
        # Clusterização usando HDBSCAN
        centroids = calculate_hdbscan_centroids_with_pca(df, embedding_column_name='pca_components')
    
    
    if centroids:
        print("Centroids created successfully")
    return df, [pca, normalizer, centroids]   
    
def return_best_context_centroids(df, query_text, centroids_parameters,n_components,distance_threshold = 0.1,k=1,show_results=False):
    
    pca, normalizer, centroids = centroids_parameters
    query_embedding = create_query_embedding(query_text)
    query_pca = create_query_pca(pca,normalizer,query_embedding)
    best_chunks_info, best_similarities = find_chunks_in_multiple_documents_euclidean_kdtree(centroids, query_pca, query_embedding, df,n_components,top_contexts = k, distance_threshold=distance_threshold,print_distances=show_results)
    if show_results:
        show_findings(best_chunks_info, best_similarities)
    return best_chunks_info


def test_brquad_centroids(df, method='document', similarity_threshold=5, n_components=127, k_clusters=1, show_results=False):
    results_list = []
    df, centroids_parameters = configure_clustering(df, method, n_components)
    
    for index, row in df.iterrows():
        st_time = time.time()
        query_text = row['query']
        df_answer = row['answer']
        best_contexts = return_best_context_centroids(df, query_text, centroids_parameters, n_components,similarity_threshold, k_clusters, show_results)
        best_contexts_text = [chunk['chunk_text'] for chunk in best_contexts]
        llm_answer = ask_with_openai(query_text, best_contexts_text)
        end_time = time.time()
        total_time = end_time - st_time
        results = {'question': query_text, 'df_answer': df_answer, 'llm_answer': llm_answer,'total_time':total_time}
        print(results)
        results_list.append(results)
        

    results_df = pd.DataFrame(results_list)
    return results_df



def save_log_df(df, method='document', similarity_threshold=0.6, n_components=127, k_clusters=1, show_results=False):
    start_time = time.time()
    results_df = test_brquad_centroids(df, method, similarity_threshold, n_components, k_clusters, show_results)
    end_time = time.time()
    total_time = end_time - start_time
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'__KDTREE__{method.upper()}_____datetime_{now}_____threshold_{similarity_threshold}_____npca_{n_components}_____k_{k_clusters}_____execution_time_{total_time:.2f}.pkl'
    results_df.to_pickle(filename)
    print(f'Tempo total de execução: {total_time:.2f} segundos')
    print(f'Arquivo salvo como: {filename}')

if __name__ == '__main__':
    original_df = pd.read_pickle('datasets/brquad-gte-dev-v2.0_embedded_with_document_names_and_answers.pkl')
    original_df.rename(columns={'document name': 'document_name', 'context': 'chunk_text'}, inplace=True)

    # Lista de configurações de clusterização para executar
    configurations = [
        #{'method': 'hdbscan', 'similarity_threshold': 0.3, 'n_components': 500, 'k_clusters': 3, 'show_results': False},
        #{'method': 'hdbscan', 'similarity_threshold': 0.4, 'n_components': 128, 'k_clusters': 3},
        #{'method': 'hdbscan', 'similarity_threshold': 0.6, 'n_components': 72, 'k_clusters': 1},
        {'method': 'hdbscan', 'similarity_threshold': 0.4, 'n_components': 72, 'k_clusters': 5},
        {'method': 'hdbscan', 'similarity_threshold': 0.6, 'n_components': 50, 'k_clusters': 3},
        {'method': 'hdbscan', 'similarity_threshold': 0.6, 'n_components': 300, 'k_clusters': 1},
        {'method': 'hdbscan', 'similarity_threshold': 0.6, 'n_components': 300, 'k_clusters': 3},
        {'method': 'hdbscan', 'similarity_threshold': 0.4, 'n_components': 500, 'k_clusters': 3},
        {'method': 'hdbscan', 'similarity_threshold': 0.4, 'n_components': 250, 'k_clusters': 3},
        {'method': 'hdbscan', 'similarity_threshold': 0.3, 'n_components': 500, 'k_clusters': 5},
        {'method': 'document', 'similarity_threshold': 0.4, 'n_components': 100, 'k_clusters': 10},
        {'method': 'document', 'similarity_threshold': 0.6, 'n_components': 128, 'k_clusters': 3},
        {'method': 'document', 'similarity_threshold': 0.4, 'n_components': 300, 'k_clusters': 3},
        {'method': 'document', 'similarity_threshold': 0.5, 'n_components': 500, 'k_clusters': 3},
        {'method': 'document', 'similarity_threshold': 0.4, 'n_components': 250, 'k_clusters': 3},
        {'method': 'document', 'similarity_threshold': 0.6, 'n_components': 500, 'k_clusters': 5}
    ]

    # Executando cada configuração com uma cópia limpa do DataFrame
    #for config in configurations:
    #    df_filtered = original_df.loc[original_df['answer'] != 'N/A'].reset_index(drop=True)
    #    save_log_df(df_filtered, **config)
    #save_log_df(original_df, method='hdbscan', similarity_threshold=0.3, n_components=500, k_clusters=3, show_results=False)
    save_log_df(original_df, method='hdbscan', similarity_threshold=0.3, n_components=300, k_clusters=5, show_results=False)


    