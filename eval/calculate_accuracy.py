import os
import pandas as pd

def calculate_accuracy(csv_path):
    df = pd.read_csv(csv_path)
    accuracy = df['is_similar'].mean()  # Calculate the mean of the is_similar column
    return accuracy

def aggregate_accuracies(results_folder, output_csv):
    accuracies = []
    filenames = []

    # Iterate over each file in the results folder
    for file in os.listdir(results_folder):
        if file.endswith('.csv'):  # Check if the file is a CSV
            csv_path = os.path.join(results_folder, file)
            accuracy = calculate_accuracy(csv_path)
            accuracies.append(accuracy)
            filenames.append(file)
            print(f"Processed {file}: Accuracy = {accuracy}")

    # Save the results to a new CSV file
    accuracy_df = pd.DataFrame({
        'Filename': filenames,
        'Accuracy': accuracies
    })
    accuracy_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    results_folder = './results'
    output_csv = './results/accuracy_summary.csv'
    aggregate_accuracies(results_folder, output_csv)
