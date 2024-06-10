# chunk_filter.py

import matplotlib.pyplot as plt

class SimilaritiesFiltering:
    def __init__(self, similarities):
        self.similarities = similarities
        self.normalized_similarities = self.normalize(similarities)
        self.gaps = self.calculate_gaps(self.normalized_similarities)
        self.chosen_chunk = self.determine_chosen_chunk(self.gaps)
    
    def normalize(self, values):
        """Normalize a list of values to be between 0 and 1."""
        min_val = min(values)
        max_val = max(values)
        return [(val - min_val) / (max_val - min_val) for val in values]

    def calculate_gaps(self, normalized_values):
        """Calculate the biggest gaps in the normalized similarity values."""
        gaps = []
        for i in range(len(normalized_values) - 1):
            if (normalized_values[i] > 0.4 and normalized_values[i + 1] >= 0.25) or (normalized_values[i] >= 0.25 and normalized_values[i + 1] > 0.4):
                gap = abs(normalized_values[i] - normalized_values[i + 1])
                gaps.append((gap, i))
        # Sort gaps by the gap value in descending order and take the top 5
        biggest_gaps = sorted(gaps, key=lambda x: x[0], reverse=True)[:5]
        return [{gap: index} for gap, index in biggest_gaps]

    def determine_chosen_chunk(self, gaps):
        """Determine the chosen chunk based on the gap logic."""
        if len(gaps) < 2:
            return None  # Not enough gaps to compare

        first_gap_value, first_gap_index = list(gaps[0].items())[0]
        second_gap_value, second_gap_index = list(gaps[1].items())[0]

        # Handle case where the difference between the biggest and second biggest gaps is less than 0.01
        if abs(first_gap_value - second_gap_value) < 0.01:
            chosen_gap = gaps[0] if first_gap_index > second_gap_index else gaps[1]
        else:
            chosen_gap = gaps[0] if first_gap_value - second_gap_value > 0.06 else gaps[1]

        return chosen_gap
    
    def plot_element(self, title="Similarity Plot"):
        """Plot the similarities and gaps."""
        similarities = self.normalized_similarities
        x_values = range(len(similarities))
        
        plt.figure(figsize=(12, 8))
        
        # Plot all similarities
        plt.bar(x_values, similarities, color='steelblue')
        
        # Plot gaps with different colors for the biggest, second biggest, third biggest, fourth biggest, and fifth biggest gaps
        gap_colors = ['red', 'orange', 'purple', 'blue', 'green']
        gap_labels = ['Biggest Gap', 'Second Biggest Gap', 'Third Biggest Gap', 'Fourth Biggest Gap', 'Fifth Biggest Gap']
        plotted_labels = set()
        
        for idx, gap in enumerate(self.gaps):
            for gap_value, gap_index in gap.items():
                y_start = max(similarities[gap_index], similarities[gap_index + 1])
                y_end = min(similarities[gap_index], similarities[gap_index + 1])
                label = f'{gap_labels[idx]}: {gap_value:.2f}' if gap_labels[idx] not in plotted_labels else None
                line_style = '-' if gap == self.chosen_chunk else '--'
                line_width = 5 if gap == self.chosen_chunk else 3
                plt.plot([gap_index + 0.5, gap_index + 0.5], [y_start, y_end], color=gap_colors[idx], linestyle=line_style, linewidth=line_width, label=label)
                plt.plot([gap_index + 0.4, gap_index + 0.6], [y_start, y_start], color=gap_colors[idx], linestyle=line_style, linewidth=2)
                plt.plot([gap_index + 0.4, gap_index + 0.6], [y_end, y_end], color=gap_colors[idx], linestyle=line_style, linewidth=2)
                if label:
                    plotted_labels.add(gap_labels[idx])
        
        plt.xlabel('ID')
        plt.ylabel('Normalized Similarity Value')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        
        plt.show()
    
    def get_selected_similarities(self):
        """Return the selected similarities (non-normalized) up to the chosen chunk."""
        if not self.chosen_chunk:
            return None
        
        chosen_gap_value, chosen_gap_index = list(self.chosen_chunk.items())[0]
        return self.similarities[:chosen_gap_index + 1]

# Example usage
if __name__ == "__main__":
    similarities_example = [
        0.3231188101086615,
        0.25219453949647763,
        0.2306857286991998,
        0.22410798219976658,
        0.16187295577197358,
        0.02294674749764869,
        -0.05971262055051674,
        -0.08868464706955437,
        -0.16515380161804116,
        -0.16994131045993055,
        -0.19854580224189622,
        -0.20284784813625012,
        -0.2035214040498723,
        -0.20547262938073163,
        -0.20942164153139675,
        -0.21846965356177384,
        -0.23006103352256924,
        -0.23027455581612033,
        -0.23826793335750174,
        -0.238989784031568,
        -0.2425193100119689,
        -0.2431802612950451,
        -0.26703432840715413,
        -0.2692219706661447,
        -0.2857713249908156
    ]
    similarities_filter = SimilaritiesFiltering(similarities_example)
    chosen_chunk = similarities_filter.chosen_chunk
    print(f"Chosen Chunk: {chosen_chunk}")
    
    similarities_filter.plot_element(title="Example Similarity Plot")
    
    selected_similarities = similarities_filter.get_selected_similarities()
    print(f"Selected Similarities: {selected_similarities}")
