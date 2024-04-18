import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Load the dataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np

def create_pca(df, n_components=2, embedding_column_name='embedding'):
    # Convert list of embeddings into a 2D numpy array
    X = np.array(df[embedding_column_name].tolist())

    # Normalize embeddings to have unit norm
    normalizer = Normalizer().fit(X)  # Fit normalizer to the embeddings
    X_normalized = normalizer.transform(X)  # Normalize embeddings

    # Perform PCA to reduce the embeddings
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_normalized)  # Apply PCA to normalized data

    # Add the principal components as a list in a single new column
    #df['pca_components'] = list(principalComponents)
    df.loc[:, 'pca_components'] = list(principalComponents)


    return pca, normalizer, n_components, df


def create_centroids(df, embedding_column_name='pca_components'):
    centroids = {}
    # Get unique document names from the DataFrame
    unique_documents = df['document_name'].unique()
    
    # Calculate the centroid for each document in the higher-dimensional PCA space
    for document in unique_documents:
        document_df = df[df['document_name'] == document]
        
        # Ensure the document_df contains the PCA components column
        if embedding_column_name in document_df:
            # Calculate the centroid as the mean of the PCA components
            centroid = np.mean(np.vstack(document_df[embedding_column_name].values), axis=0)
            centroids[document] = centroid  # Store the centroid
        else:
            print(f"Missing PCA components for {document}. Make sure that PCA was applied and '{embedding_column_name}' column exists.")

    return centroids











