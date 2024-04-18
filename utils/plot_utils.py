import pandas as pd
from termcolor import colored
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go



def show_findings(best_chunks_info, best_similarities):


    # Print the information
    header = colored(f"{'Best Chunk Infos':^60}", 'blue', 'on_white', attrs=['bold', 'underline'])  # Header styling
    divider = colored('-' * 60, 'red')  # Divider styling

    # Print header
    print(header)

    # Loop through each of the best chunks and their corresponding similarity
    for index, (chunk_info, similarity) in enumerate(zip(best_chunks_info, best_similarities), start=1):
        # Clean the chunk text to remove tab characters
        
        print(f"Chunk #{index}:")
        clean_chunk_text = chunk_info['chunk_text'].replace('\t', ' ')
        similarity_row = colored(f"{'Best Similarity:':<20}", 'yellow') + str(similarity)
        print(similarity_row)
        document_row = colored(f"{'Document Name:':<20}", 'green') + chunk_info['document_name']
        print(document_row)
        try:
            cluster_row = colored(f"{'Cluster:':<20}", 'blue') + str(chunk_info['cluster_label'])
            print(cluster_row)
            init_page_row = colored(f"{'Initial Page:':<20}", 'magenta') + str(chunk_info['init_page'])
            end_page_row = colored(f"{'End Page:':<20}", 'magenta') + str(chunk_info['end_page'])
        except KeyError:
            cluster_row = colored(f"{'Cluster:':<20}", 'blue') + str(chunk_info['document_name'])
            print(cluster_row)
        text_row = colored(f"{'Chunk Text:':<20}", 'cyan') + clean_chunk_text + "..."  # Truncated text for display


        # Print individual chunk details
        
        
        
        
        print(text_row)
        

    
def plot_clusters_2_components(df,plot=False):

    # Plot initialization
    fig = go.Figure()
    colors = {
        'Harry Potter e a Pedra Filosofal.pdf': 'red',
        'Harry Potter e a Câmara Secreta.pdf': 'green',
        'Harry Potter e o prisioneiro de Azkaban.pdf': 'blue',
        'Harry Potter e o Cálice de Fogo.pdf': 'yellow',
        'Harry Potter e a Ordem da Fênix.pdf': 'cyan',
        'Harry Potter e o Enigma do Príncipe.pdf': 'magenta',
        'Harry Potter e as Relíquias da Morte.pdf': 'grey'
    }
    centroids = {}

    # Add traces for each book
    for book, color in colors.items():
        book_df = df[df['document_name'] == book]
        fig.add_trace(go.Scatter(
            x=book_df['principal component 1'], 
            y=book_df['principal component 2'],
            text=book_df['first_15_words'],  # Display on hover
            mode='markers',
            marker_color=color,
            name=book
        ))
        book_df = df[df['document_name'] == book]
        centroid = book_df[['principal component 1', 'principal component 2']].mean().values
        centroids[book] = centroid  # Store the centroid
        fig.add_trace(go.Scatter(
            x=[centroid[0]], 
            y=[centroid[1]],
            text=[book],  # Book name for centroid hover
            mode='markers+text',
            marker_symbol='x',
            marker_size=12,
            marker_color='black',
            showlegend=False,
            textposition="top center"
        ))

    # Customize layout
    fig.update_layout(
        title='PCA of Book Embeddings with Centroids',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        legend_title='Book Name'
    )

    # Show plot
    if plot:
        fig.show()
    
    return df,fig,centroids
