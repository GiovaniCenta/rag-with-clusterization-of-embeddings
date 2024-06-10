import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import plotly.figure_factory as ff

def apply_hierarchical_clustering(data, indexes, n_clusters=None, method='ward'):
    # Prepare data for clustering: extract the PCA components
    X_pca = np.vstack(data[:, indexes['pca_components']])
    
    # Apply Hierarchical Clustering using the specified linkage method
    Z = linkage(X_pca, method=method)
    
    # Determine cluster labels
    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion='maxclust')
    else:
        # Assuming a default behavior, cut by distance not by the number of clusters
        labels = fcluster(Z, 8, criterion='distance')
    
    # Assign cluster labels back to the data array
    for i in range(data.shape[0]):
        data[i, indexes['cluster_label']] = labels[i]

    return Z, data

def calculate_centroids_hc(data, indexes, plot_graph=False):
    Z, data = apply_hierarchical_clustering(data, indexes)
    cluster_labels = data[:, indexes['cluster_label']]
    unique_clusters = np.unique(cluster_labels)
    
    centroids = {}
    centroid_document_map = {}
    
    # Process normal clusters, excluding noise
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise
        cluster_mask = cluster_labels == cluster
        cluster_data = data[cluster_mask]
        
        # Calculate the centroid from PCA components
        if cluster_data.size > 0:
            pca_vectors = np.vstack(cluster_data[:, indexes['pca_components']])
            centroid = pca_vectors.mean(axis=0)
            centroids[cluster] = centroid
            centroid_document_map[cluster] = (cluster, centroid)

    # Optionally plot the dendrogram
    if plot_graph:
        labels = data[:, indexes['document_name']]  # Use document names as labels
        plot_dendrogram(Z, labels)
        exit(8)

    return centroids, centroid_document_map

import numpy as np
import plotly.figure_factory as ff

def plot_dendrogram(Z, labels):
    # Ensure the labels length matches the number of initial observations in Z + 1
    if len(labels) != len(Z):
        print("Adding a dummy linkage to match the number of labels")
        # Create a dummy linkage that extends the last linkage
        last_linkage = Z[-1, :]
        dummy_linkage = last_linkage.copy()
        dummy_linkage[2] += 1  # Increase the distance slightly
        Z = np.vstack([Z, dummy_linkage])  # Append the dummy linkage
    
    # Convert all labels to string for consistency in the plot
    labels = [str(label) for label in labels]

    # Plotting the dendrogram with Plotly
    fig = ff.create_dendrogram(Z, orientation='left', labels=labels)
    fig.update_layout(
        width=800,
        height=800,
        title='Hierarchical Clustering Dendrogram',
        xaxis_title='Euclidean distances',
        yaxis_title='Data points or (Cluster size)',
        xaxis=dict(tickmode='array'),
        yaxis=dict(ticks='outside', ticklen=8, tickwidth=2, tickcolor='#000')
    )
    fig.show()

# Remember to call this function with the correct labels
# plot_dendrogram(Z, your_labels_list)

