import time
from cluster_methods.create_clusters import ClusteringCreation

from utils.openai_utils import *
from utils.load_data import *
from find_chunk_methods.similarities_filter import *
from cluster_methods.hdbscan_methods import *

from cluster_methods.hclustering_methods import *
from cluster_methods.knn_methods import *
from cluster_methods.create_clusters import ClusteringCreation
from find_chunk_methods.find_closest_cluster_to_query import ClosestClustersToQuery
from find_chunk_methods.find_chunks_inside_cluster import FindChunksInsideCluster
from utils.ask_with_contexts import AskWithContexts
import time
import json


def run_clusters(data_context, query_text, indexes_context, 
                                 method='document',inside_thr=0.08, similarity_threshold=5, 
                                 n_components=127,show_results=False, show_clusters=False,show_clusters_distances = False):
   
    #Creating centroids and clusters
    ClusteringCreation_module = ClusteringCreation(data_context, indexes_context, method, n_components, 
                                                   show_clusters=show_clusters)
    data,centroids_parameters = ClusteringCreation_module.create_clusters()
    

    print("Successfully created clustering.")
    

    ClosestClustersToQuery_module = ClosestClustersToQuery(data, indexes_context, query_text, n_components,centroids_parameters,
                                                           show_clusters_distances=show_clusters_distances)
    closest_documents,query_pca = ClosestClustersToQuery_module.find_closest_clusters()
    
    print("Successfully found closest clusters.")
    
    
    FindChunksInsideCluster_module = FindChunksInsideCluster(indexes_context, query_pca, closest_documents, data,
                                                             inside_cluster_thr=0.08,llm_token_limit=16000)
    best_contexts,best_similarities = FindChunksInsideCluster_module.find_best_chunks()
    
    
    print("Successfully found best chunks.")
    
    AskWithContexts_module = AskWithContexts(query_text, best_contexts, indexes_context, 
                                             llm_token_limit=16000,show_results=show_results)
    results_df = AskWithContexts_module.ask_with_filtered_contexts()
    
    print("Successfully asked with filtered contexts.")
    
    return results_df

if __name__ == "__main__":
    data_context = np.load('datasets/dataset.npy',allow_pickle=True)
    indexes_context = {'chunk_text': 0, 'document_name':1, 'embedding': 2,'cluster_label':3,'pca_components':4}
    n_components = len(data_context)
    query_text = "What is Carlos Silva CPF and RG?"
    results_df = run_clusters(data_context, query_text, indexes_context,method='kmeans',
                              show_results=True,show_clusters_distances=True,show_clusters=False)
    print("Answer: ", results_df['llm_answer'].values[0])
