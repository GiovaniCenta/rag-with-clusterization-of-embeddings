import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('results/sim_log.json', 'r') as f:
    data = json.load(f)

# Ensure the output directories exist
output_dir = 'eval/eval_results/sim_plots'
flag_analysis_output_dir = 'eval'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(flag_analysis_output_dir, exist_ok=True)

# Function to sanitize the question string for file naming
def sanitize_filename(text, max_length=50):
    sanitized = "".join(c if c.isalnum() else "_" for c in text)
    return sanitized[:max_length]

# Function to normalize values between 0 and 1
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) for val in values]

# Function to calculate the biggest gaps in the normalized similarity values
def calculate_gaps(normalized_values):
    gaps = []
    for i in range(len(normalized_values) - 1):
        if (normalized_values[i] > 0.4 and normalized_values[i + 1] >= 0.25) or (normalized_values[i] >= 0.25 and normalized_values[i + 1] > 0.4):
            gap = abs(normalized_values[i] - normalized_values[i + 1])
            gaps.append((gap, i))
    # Sort gaps by the gap value in descending order and take the top 5
    biggest_gaps = sorted(gaps, key=lambda x: x[0], reverse=True)[:5]
    return [{gap: index} for gap, index in biggest_gaps]

# Function to determine the chosen chunk based on the gap logic
def determine_chosen_chunk(gaps):
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

# Function to plot each element
def plot_element(element, index):
    similarities = element['similarities_vector']
    x_values = range(len(similarities))
    chunks_needed = element['chunks_needed']
    
    # Define prettier colors
    colors = ['mediumseagreen'] * chunks_needed + ['steelblue'] * (len(similarities) - chunks_needed)
    
    plt.figure(figsize=(12, 8))
    
    # Plot all similarities with specified colors
    plt.bar(x_values, similarities, color=colors[:len(similarities)])
    
    # Plot gaps with different colors for the biggest, second biggest, third biggest, fourth biggest, and fifth biggest gaps
    gap_colors = ['red', 'orange', 'purple', 'blue', 'green']
    gap_labels = ['Biggest Gap', 'Second Biggest Gap', 'Third Biggest Gap', 'Fourth Biggest Gap', 'Fifth Biggest Gap']
    plotted_labels = set()
    
    chosen_chunk = element.get('chosen_chunk')
    for idx, gap in enumerate(element['gaps']):
        for gap_value, gap_index in gap.items():
            y_start = max(similarities[gap_index], similarities[gap_index + 1])
            y_end = min(similarities[gap_index], similarities[gap_index + 1])
            label = f'{gap_labels[idx]}: {gap_value:.2f}' if gap_labels[idx] not in plotted_labels else None
            line_style = '-' if gap == chosen_chunk else '--'
            line_width = 5 if gap == chosen_chunk else 3
            plt.plot([gap_index + 0.5, gap_index + 0.5], [y_start, y_end], color=gap_colors[idx], linestyle=line_style, linewidth=line_width, label=label)
            plt.plot([gap_index + 0.4, gap_index + 0.6], [y_start, y_start], color=gap_colors[idx], linestyle=line_style, linewidth=2)
            plt.plot([gap_index + 0.4, gap_index + 0.6], [y_end, y_end], color=gap_colors[idx], linestyle=line_style, linewidth=2)
            if label:
                plotted_labels.add(gap_labels[idx])
    
    plt.xlabel('ID')
    plt.ylabel('Normalized Similarity Value')
    plt.title(f"Question: {element['question']}\nAnswer: {element['df_answer']}\nChunks Needed: {chunks_needed}\nChosen Chunk: {chosen_chunk}")
    plt.grid(True)
    plt.legend()
    
    # Sanitize the question for use in the filename
    sanitized_question = sanitize_filename(element['question'])
    
    # Save the plot
    plt.savefig(f'{output_dir}/plot_idx_{index}___question_{sanitized_question}.png')
    plt.close()

# Process each element to normalize similarities and calculate gaps
for element in data:
    similarities = element['similarities_vector']
    normalized_similarities = normalize(similarities)
    element['similarities_vector'] = normalized_similarities
    element['gaps'] = calculate_gaps(normalized_similarities)
    element['chosen_chunk'] = determine_chosen_chunk(element['gaps'])

# Analyze gaps and add flag_gap
for element in data:
    chunks_needed_idx = element['chunks_needed'] - 1
    element['flag_gap'] = 0  # Default value
    if 'gaps' in element:
        for i, gap in enumerate(element['gaps']):
            for gap_value, gap_index in gap.items():
                if gap_index == chunks_needed_idx:
                    element['flag_gap'] = i + 1  # 1 for biggest gap, 2 for second biggest, 3 for third biggest, 4 for fourth biggest, 5 for fifth biggest

# Save the updated data to a new JSON file
output_file = 'results/sim_log_gaps.json'
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

# Create a new JSON for flag analysis
flag_analysis_data = [
    {
        "question": element['question'],
        "chunks_needed": element['chunks_needed'],
        "gaps": element['gaps'],
        "flag_gap": element['flag_gap'],
        "chosen_chunk": element['chosen_chunk']
    }
    for element in data
]

# Save the flag analysis data to a new JSON file
flag_analysis_output_file = os.path.join(flag_analysis_output_dir, 'flag_analysis.json')
with open(flag_analysis_output_file, 'w') as f:
    json.dump(flag_analysis_data, f, indent=4)

# Iterate through each element in the JSON and plot
for i, element in enumerate(data):
    plot_element(element, i)

print(f"Updated data has been saved to {output_file}")
print(f"Flag analysis data has been saved to {flag_analysis_output_file}")
print("Plots have been generated and saved.")
