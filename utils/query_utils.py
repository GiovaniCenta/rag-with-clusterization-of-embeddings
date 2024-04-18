import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from utils.openai_utils import embed_with_openai


def create_query_embedding(query_text):
    query_embedding = embed_with_openai(query_text)  # Get embedding using OpenAI
    return query_embedding

def create_query_pca(pca,scaler,query_embedding):
    query_embedding_scaled = scaler.transform(np.array(query_embedding).reshape(1, -1))  # Scale using the same scaler
    query_pca = pca.transform(query_embedding_scaled)  # Apply PCA
    return query_pca[0]   #consistency with the other functions

def plot_query_2_components(query_pca,fig,query_text,plot=False):


    # Add the query to the plot for visualization
    fig.add_trace(go.Scatter(
        x=[query_pca[0]], 
        y=[query_pca[1]],
        text=query_text,  # Display the actual query text
        mode='markers+text',
        marker_symbol='star',  # Change marker symbol for better visibility
        marker_size=20,  # Increase marker size
        marker_color="lime",  # Use a bright color for visibility
        showlegend=False,
        textposition="top center",  # Adjust text position for clarity
        textfont=dict(  # Customize font properties for better visibility
            family="Arial, sans-serif",
            size=12,  # Increase text size
            color="blue",  # Choose a color that contrasts well with the marker color
        ),
    ))
    if plot:
        fig.show()
    
    

