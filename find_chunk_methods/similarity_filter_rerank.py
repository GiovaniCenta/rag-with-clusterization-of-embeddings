from transformers import pipeline

class SimilaritiesFilteringWithReranker:
    def __init__(self, reranker, query_text, chunk_texts):
        self.reranker = reranker
        self.query_text = query_text
        self.chunk_texts = chunk_texts
        self.similarities = self.calculate_reranked_similarities()
        self.chosen_chunks = self.determine_chosen_chunks()
        self.log_similarities()

        print(f"Chosen Chunks: {self.chosen_chunks}")

    def calculate_reranked_similarities(self):
        """Calculate similarities using the reranker."""
        inputs = [{"text": self.query_text, "text_pair": chunk} for chunk in self.chunk_texts]
        results = self.reranker(inputs, return_all_scores=False)

        # Extract scores and return them with the corresponding chunk texts
        similarities = [(result['score'], chunk) for result, chunk in zip(results, self.chunk_texts)]
        # Sort by similarity in descending order
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities

    def determine_chosen_chunks(self):
        """Determine the chosen chunks based on the reranked similarities."""
        # Filter chunks with similarity scores above a threshold, e.g., 0.05
        threshold = 0.01
        chosen_chunks = [(sim, chunk) for sim, chunk in self.similarities if sim >= threshold]
        return chosen_chunks

    def log_similarities(self):
        """Log all similarities with the corresponding chunk texts to a file."""
        with open("similarities_debug.txt", "a") as debug_file:  # Open file in append mode
            debug_file.write("- - - - - - - - - -- - - - - - - - - - - -\n")
            debug_file.write(f"Query: {self.query_text}\n")
            debug_file.write("Reranked Similarity: || Chunk:\n")
            for similarity, chunk_text in self.similarities:
                debug_file.write(f"{similarity:.4f}: || {chunk_text}\n")
            debug_file.write("- - - - - - - - - - - - - -- - - - - - - -\n")

    def get_selected_similarities(self):
        """Return the selected similarities based on the reranker."""
        return [sim for sim, chunk in self.chosen_chunks]

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = "local_models/bge-reranker-v2-m3"
    tokenizer_path = "local_models/bge-reranker-v2-m3"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    reranker = pipeline("text-classification", model=model, tokenizer=tokenizer)

    chunk_texts_example = [
        "Chunk text 1 about the meaning of life",
        "Chunk text 2 about different cultures and ideologies",
        "Chunk text 3 about intrinsic meaning and subjective experience",
        "Chunk text 4 about philosophical views",
        "Chunk text 5 about religious context",
        "Chunk text 6 about scientific perspective",
        "Chunk text 7 about modern science",
        "Chunk text 8 about self-discovery",
        "Chunk text 9 about human experience",
        "Chunk text 10 about the quest for knowledge",
    ]

    query_text_example = "What is the meaning of life?"
    similarities_filter = SimilaritiesFilteringWithReranker(reranker, query_text_example, chunk_texts_example)
    chosen_chunks = similarities_filter.chosen_chunks
    print(f"Chosen Chunks: {chosen_chunks}")

    selected_similarities = similarities_filter.get_selected_similarities()
    print(f"Selected Similarities: {selected_similarities}")
