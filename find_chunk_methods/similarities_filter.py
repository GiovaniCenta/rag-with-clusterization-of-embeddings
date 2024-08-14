import matplotlib.pyplot as plt

class SimilaritiesFiltering:
    def __init__(self, similarities, query_text, chunk_texts):
        self.similarities = similarities
        self.query_text = query_text
        self.chunk_texts = chunk_texts
        self.chosen_chunks = self.determine_chosen_chunks()
        self.log_similarities()

        print(f"Chosen Chunks: {self.chosen_chunks}")

    def determine_chosen_chunks(self):
        """Determine the chosen chunks based on the new condition."""
        chosen_chunks = []
        for i in range(len(self.similarities) - 1):
            current_sim = self.similarities[i]
            next_sim = self.similarities[i + 1]
            gap = abs(current_sim - next_sim)

            # Check the condition
            if current_sim > 0.25 and next_sim < 0.1 and gap > 0.2:
                chosen_chunks.append((current_sim, self.chunk_texts[i]))
                break  # Stop adding chunks once the condition is met
            else:
                chosen_chunks.append((current_sim, self.chunk_texts[i]))

        # Add the last chunk if it's above the threshold
        if self.similarities[-1] > 0.05:
            chosen_chunks.append((self.similarities[-1], self.chunk_texts[-1]))

        return chosen_chunks

    def log_similarities(self):
        """Log all similarities with the corresponding chunk texts to a file."""
        with open("similarities_debug.txt", "a") as debug_file:  # Open file in append mode
            debug_file.write("- - - - - - - - - -- - - - - - - - - - - -\n")
            debug_file.write(f"Query: {self.query_text}\n")
            debug_file.write("Similarity: || Chunk:\n")
            for similarity, chunk_text in zip(self.similarities, self.chunk_texts):
                debug_file.write(f"{similarity:.4f}: || {chunk_text}\n")
            debug_file.write("- - - - - - - - - - - - - -- - - - - - - -\n")

    def plot_element(self, title="Similarity Plot"):
        """Plot the similarities."""
        similarities = self.similarities
        x_values = range(len(similarities))

        plt.figure(figsize=(12, 8))

        # Plot all similarities
        plt.bar(x_values, similarities, color='steelblue')

        plt.xlabel('ID')
        plt.ylabel('Similarity Value')
        plt.title(title)
        plt.grid(True)
        plt.show()

    def get_selected_similarities(self):
        """Return the selected similarities (non-normalized) based on the chosen chunks."""
        return [sim for sim, chunk in self.chosen_chunks]

# Example usage
if __name__ == "__main__":
    similarities_example = [
        0.3895260697348626,
        0.3458771785342579,
        0.2927317794188812,
        0.06846817900692459,
        0.05100010509351627
    ]

    chunk_texts_example = [
        "Machu Picchu was constructed in the 15th century by the Inca Empire and served as a royal estate or religious site for the Inca leaders",
        "Machu Picchu was designated as a UNESCO World Heritage Site in 1983, acknowledging its cultural significance and the need to preserve it for future generations",
        "The Inca Empire, known for its advanced engineering and architecture, built Machu Picchu high in the Andes Mountains, demonstrating their ability to adapt to challenging environments",
        "The Seven Wonders of the Ancient World were remarkable constructions known for their grandeur and engineering, with the Great Pyramid of Giza being the only one still standing today",
        "The most well-preserved sections of the Great Wall of China, the monumental defensive structure, were built during the Ming Dynasty, which expanded and fortified the wall"
    ]

    query_text_example = "When was Machu Picchu built, which civilization constructed it, who rediscovered it, and what is its status today?"
    similarities_filter = SimilaritiesFiltering(similarities_example, query_text_example, chunk_texts_example)
    chosen_chunks = similarities_filter.chosen_chunks
    print(f"Chosen Chunks: {chosen_chunks}")

    similarities_filter.plot_element(title="Example Similarity Plot")

    selected_similarities = similarities_filter.get_selected_similarities()
    print(f"Selected Similarities: {selected_similarities}")
