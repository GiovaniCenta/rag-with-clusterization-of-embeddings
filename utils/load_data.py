

import numpy as np
import json
from utils.openai_utils import *

def create_sets(data_json,qa_json,indexes_context,indexes_qa):
    if data_json is None:
        pass
    else:
        data_context = json_to_numpy(data_json,indexes_context)
        for i in range(data_context.shape[0]):
            print("Processing chunk ",i," of ",data_context.shape[0])
            data_context[i, indexes_context['embedding']] = embbed_with_ai_gateway(data_context[i, indexes_context['chunk_text']])
        np.save('datasets/dataset.npy',data_context)
        print("Dataset contexts saved in datasets/dataset.npy")
    
    data_qa = load_qa_data_to_numpy(qa_json,indexes_qa)
    for i in range(data_qa.shape[0]):
        print("Processing chunk ",i," of ",data_qa.shape[0])
        data_qa[i, indexes_qa['embedding']] = embbed_with_ai_gateway(data_qa[i, indexes_qa['question']])    
    np.save('datasets/qa-papers.npy',data_qa)
    print("Dataset QA saved in datasets/qa-papers.npy")


def load_qa_data_to_numpy(filename,indexes_qa):
    # Load the JSON data from a file
    with open(filename, 'r') as file:
        data = json.load(file)
    # Initialize an empty list to hold the data
    numpy_data = []

    # Process each entry in the JSON data
    for entry in data:
        # Extract question, answer, and title
        question = entry["question"]
        answer = entry["answers"]["text"]
        title = entry["title"]
        
        # Initialize placeholders for embedding and pca_components
        embedding = None  # Placeholder, since we don't have the embeddings yet
        pca_components = None  # Placeholder for PCA components
        
        # Append the data to the list, respecting the order defined in indexes_qa
        row = [None] * len(indexes_qa)
        row[indexes_qa['question']] = question
        row[indexes_qa['answer']] = answer
        row[indexes_qa['document_name']] = title
        row[indexes_qa['embedding']] = embedding
        row[indexes_qa['pca_components']] = pca_components
        
        numpy_data.append(row)

    # Convert the list to a NumPy array
    numpy_array = np.array(numpy_data, dtype=object)

    return numpy_array

def json_to_numpy(json_file,indexes_context):
    # Load your JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    numpy_data = []

    # Loop through each entry in the JSON data
    for entry in data:
        # Extract chunk_text and document_name
        chunk_text = entry["chunk_text"]
        document_name = entry["title"]
        
        # Initialize placeholders for embedding, cluster_label, and pca_components
        embedding = None  # Placeholder, since we don't have the embeddings yet
        cluster_label = None  # Placeholder for cluster labels
        pca_components = None  # Placeholder for PCA components
        
        # Append the data to the list, respecting the order defined in indexes_context
        row = [None] * len(indexes_context)
        row[indexes_context['chunk_text']] = chunk_text
        row[indexes_context['document_name']] = document_name
        row[indexes_context['embedding']] = embedding
        row[indexes_context['cluster_label']] = cluster_label
        row[indexes_context['pca_components']] = pca_components
        
        numpy_data.append(row)

    # Convert the list to a NumPy array
    numpy_array = np.array(numpy_data, dtype=object)
    
    return numpy_array