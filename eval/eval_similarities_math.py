import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('results/sim_log.json', 'r') as f:
    data = json.load(f)

# Ensure the output directory exists
output_dir = 'eval/eval_results/sim_plots'
os.makedirs(output_dir, exist_ok=True)

# Function to sanitize the question string for file naming
def sanitize_filename(text, max_length=50):
    sanitized = "".join(c if c.isalnum() else "_" for c in text)
    return sanitized[:max_length]

# Function to normalize values between 0 and 1
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) for val in values]

# Function to filter green and blue bars using polynomial fit and chunks needed
def filter_bars_with_threshold(normalized_vector, chunks_needed, degree=3):
    x = np.arange(len(normalized_vector))
    y = normalized_vector
    
    # Fit a polynomial curve
    poly_coeffs = np.polyfit(x, y, degree)
    poly_fit = np.poly1d(poly_coeffs)
    
    # Evaluate the polynomial fit at the needed chunks
    threshold_value = poly_fit(chunks_needed - 1)
    
    # Separate green and blue bars based on the threshold
    green_bars = y[:chunks_needed]
    blue_bars = y[chunks_needed:]
    
    return green_bars, blue_bars, poly_fit

# Function to plot each element
def plot_element(element, index):
    similarities = element['similarities_vector']
    normalized_similarities = normalize(similarities)
    x_values = range(len(normalized_similarities))
    chunks_needed = element['chunks_needed']
    
    # Apply the threshold filtering
    green_bars, blue_bars, polynomial_fit = filter_bars_with_threshold(normalized_similarities, chunks_needed)
    
    # Define colors for the bars
    colors = ['mediumseagreen'] * len(green_bars) + ['steelblue'] * len(blue_bars)
    
    plt.figure(figsize=(12, 8))
    
    # Plot all normalized similarities with specified colors
    plt.bar(x_values, normalized_similarities, color=colors[:len(normalized_similarities)])
    
    # Fit a polynomial curve and adjust to start at y = 1
    degree = 3  # Degree of the polynomial
    x_fit = np.linspace(0, len(normalized_similarities) - 1, 100)
    y_fit = polynomial_fit(x_fit)
    
    # Scale the polynomial curve to start at y = 1
    y_fit = (y_fit - min(y_fit)) / (max(y_fit) - min(y_fit))
    y_fit = 1 - (1 - y_fit)  # Ensure the curve starts at y = 1
    
    # Plot the polynomial curve
    plt.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
    
    plt.xlabel('ID')
    plt.ylabel('Normalized Similarity Value')
    plt.title(f"Question: {element['question']}\nChunks Needed: {chunks_needed}")
    plt.grid(True)
    plt.legend()
    
    # Sanitize the question for use in the filename
    sanitized_question = sanitize_filename(element['question'])
    
    # Save the plot
    plt.savefig(f'{output_dir}/plot_idx_{index}___question_{sanitized_question}.png')
    plt.close()

# Iterate through each element in the JSON and plot
for i, element in enumerate(data):
    plot_element(element, i)

print("Plots have been generated and saved.")
