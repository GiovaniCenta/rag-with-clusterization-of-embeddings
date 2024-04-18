import numpy as np
from sklearn.decomposition import PCA

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
from scipy.spatial.distance import cosine

def find_closest_books(query_embedding, centroids, distance_threshold=0.1,embedding_column_name = 'embedding'):
    # Calculate the cosine distances between the query and each centroid
    distances = {}
    for book, centroid in centroids.items():
        # Note: cosine() function returns similarity, 1 - cosine() will convert it to distance
        dist = 1 - cosine(query_embedding, centroid)
        distances[book] = dist
    
    # Sort the books based on their distance from the query
    sorted_distances = sorted(distances.items(), key=lambda item: item[1], reverse=True)  # cosine similarity is higher for closer items, so we reverse sort

    # Get the distance of the closest book
    closest_distance = sorted_distances[0][1] if sorted_distances else float('inf')
    
    # Find all books where the distance is within the threshold of the closest book's distance
    books_to_compare = [book for book, distance in sorted_distances if closest_distance - distance <= distance_threshold]

    return books_to_compare

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import euclidean

from scipy.spatial import KDTree


def query_to_centroid_distances(query_pca, centroids):
    distances = {}
    for book, centroid in centroids.items():
        dist = euclidean_distance(query_pca, centroid)
        distances[book] = dist
    return distances
from scipy.spatial.distance import cosine
import numpy as np

def compare_similarity(query_embedding, chunks, book_name, embedding_column_name):
    # Ensure the query embedding is a 1-D numpy array
    query_vec = np.array(query_embedding)
    
    # Dictionary to store similarities
    similarities = {}

    # Calculate cosine similarity for each chunk
    for index, row in chunks.iterrows():
        chunk_embedding = np.array(row[embedding_column_name]).flatten()  # Convert chunk embedding to 1-D numpy array
        # Compute cosine similarity (cosine returns the distance, so subtract from 1)
        cos_sim = 1 - cosine(query_vec, chunk_embedding)
        # Store the similarity along with chunk text and book name
        similarities[(row['chunk_text'], book_name)] = cos_sim

    return similarities



from scipy.spatial import KDTree

from scipy.spatial import KDTree

def find_best_chunks_kdtree(top_contexts, query_embedding, closest_books, df, embedding_column_name='pca_components'):
    all_similarities = []

    for book in closest_books:
        try:
            book_chunks = df[df['cluster_label'] == book]
        except KeyError:
            book_chunks = df[df['document_name'] == book]
        
        if book_chunks.empty:
            #print(f"Nenhum chunk encontrado para o livro/documento: {book}")
            continue  # Se não houver chunks, continua para o próximo livro/documento

        # Tentativa de construir uma matriz com todos os embeddings dos chunks
        try:
            chunk_embeddings = np.vstack(book_chunks[embedding_column_name].tolist())
        except ValueError as e:
            #print(f"Erro ao processar embeddings para o livro/documento {book}: {e}")
            continue

        # Se chunk_embeddings estiver vazio, não tenta criar a árvore
        if chunk_embeddings.size == 0:
            #print(f"Nenhum embedding válido encontrado para o livro/documento: {book}")
            continue

        # Cria o KDTree a partir dos embeddings
        tree = KDTree(chunk_embeddings)
        
        # Realiza uma consulta ao KDTree para encontrar os chunks mais próximos com base no embedding de consulta
        distances, indices = tree.query(query_embedding.reshape(1, -1), k=min(top_contexts, len(chunk_embeddings)))

        # Coleta similaridades e seus correspondentes índices no DataFrame
        for dist, idx in zip(distances[0], indices[0]):
            cos_sim = 1 - dist  # Converte a distância em uma medida de similaridade
            if idx < len(book_chunks):
                chunk_index = book_chunks.iloc[idx].name  # Acessa o índice do DataFrame do chunk de forma segura
                all_similarities.append((cos_sim, chunk_index))
            else:
                continue
                #print(f"Index {idx} out of bounds for book_chunks with length {len(book_chunks)}")

    # Ordena todas as similaridades encontradas em ordem decrescente
    all_similarities.sort(reverse=True, key=lambda x: x[0])

    # Obtém as top_contexts similaridades mais altas
    top_similarities = all_similarities[:top_contexts]

    # Recupera as informações dos melhores chunks usando seus índices
    best_chunks_info = [df.loc[idx] for _, idx in top_similarities if idx in df.index]

    return best_chunks_info, [sim for sim, _ in top_similarities]


def filter_closest_books(query_embedding, centroids, distance_threshold=0.1):
    # Calculate the Euclidean distances between the query and each centroid
    distances = {book: euclidean(query_embedding, centroid) for book, centroid in centroids.items()}

    # Sort the books based on their distance from the query
    sorted_distances = sorted(distances.items(), key=lambda item: item[1])  # Euclidean distance is lower for closer items

    # Get the distance of the closest book
    closest_distance = sorted_distances[0][1] if sorted_distances else float('inf')

    # Find all books where the distance is within the threshold of the closest book's distance
    books_within_threshold = [book for book, distance in sorted_distances if distance <= closest_distance + distance_threshold]

    return books_within_threshold
