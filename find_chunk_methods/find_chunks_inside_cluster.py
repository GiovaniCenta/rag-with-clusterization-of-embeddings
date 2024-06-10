import numpy as np
from scipy.spatial import KDTree
from find_chunk_methods.similarities_filter import SimilaritiesFiltering

class FindChunksInsideCluster:
    def __init__(self, indexes, query_pca, closest_books, data, inside_cluster_thr=0.08, llm_token_limit=16000):
        
        self.indexes = indexes
        self.query_pca = query_pca
        self.closest_books = closest_books
        self.data = data
        self.inside_cluster_thr = inside_cluster_thr
        self.llm_token_limit = llm_token_limit

    def find_best_chunks(self):
        all_similarities = []
        seen_chunks = set()  # To avoid adding the same chunk multiple times

        for book_index in self.closest_books:
            book_chunks = self.data[self.data[:, self.indexes['cluster_label']] == book_index]

            if book_chunks.size == 0:
                continue

            try:
                chunk_embeddings = np.vstack(book_chunks[:, self.indexes['pca_components']])
            except ValueError:
                continue

            if chunk_embeddings.size == 0:
                continue

            tree = KDTree(chunk_embeddings)
            distances, indices = tree.query(self.query_pca.reshape(1, -1), k=len(chunk_embeddings))

            for dist, idx in zip(distances[0], indices[0]):
                if idx in seen_chunks:
                    continue  # Skip if this chunk has already been processed

                cos_sim = 1 - dist
                seen_chunks.add(idx)
                chunk_row = book_chunks[idx]  # Access the entire row of the chunk
                all_similarities.append((cos_sim, chunk_row))

        # Call the filtering function here after collecting all similarities
        filtered_best_chunks_info, filtered_best_similarities = self.filter_chunks_by_similarity_and_token_limit(
            all_similarities
        )

        return filtered_best_chunks_info, filtered_best_similarities

    def filter_chunks_by_similarity_and_token_limit(self, all_chunks):
        # Extract similarities from all_chunks
        all_similarities = [sim for sim, chunk in all_chunks]

        # Initialize SimilaritiesFiltering with all similarities
        similarities_filter = SimilaritiesFiltering(all_similarities)

        # Get chosen chunk based on the similarity filtering logic
        chosen_chunk = similarities_filter.chosen_chunk

        if not chosen_chunk:
            return [], []  # Return empty lists if no chosen chunk

        chosen_gap_value, chosen_gap_index = list(chosen_chunk.items())[0]

        selected_chunks_info = []
        selected_similarities = []
        total_tokens = 0

        # Select chunks until the token limit is reached
        for i, (similarity, chunk_row) in enumerate(all_chunks):
            if i <= chosen_gap_index:
                chunk_text = chunk_row[self.indexes['chunk_text']]
                num_tokens = len(chunk_text.split())  # Assuming tokenization by spaces
                if total_tokens + num_tokens > self.llm_token_limit:
                    break  # Stop if adding this chunk would exceed the token limit
                selected_chunks_info.append(chunk_row)
                selected_similarities.append(similarity)
                total_tokens += num_tokens

        return selected_chunks_info, selected_similarities
