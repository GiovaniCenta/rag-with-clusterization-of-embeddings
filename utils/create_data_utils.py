import os
import pandas as pd
import openai
from dotenv import load_dotenv
import PyPDF2  # Import PyPDF2
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_methods.hdbscan_methods import *
from cluster_methods.docname_methods import *
from utils.query_utils import *
from utils.openai_utils import *

from utils.plot_utils import *
import pandas as pd

def extract_text_from_pdf(pdf_path,num_paginas = 1000):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        pages_text = []
        for page_num in range(min(num_paginas, len(reader.pages))):  # Extract from the first num_paginas pages or total pages if less than 100
            page = reader.pages[page_num]
            page_text = page.extract_text() if page.extract_text() else ''  # Extract text or return empty string if None
            pages_text.append(page_text)
    return pages_text

def split_into_chunks(pages_text, chunk_size):
    chunks = []  # To hold chunks of concatenated page texts
    page_ranges = []  # To hold corresponding page ranges for each chunk

    current_chunk = ''  # Accumulates text for the current chunk
    chunk_start_page = 0  # Tracks the start page index of the current chunk
    for i, page_text in enumerate(pages_text):
        # Check if the current page fits into the current chunk
        if len(current_chunk) + len(page_text) <= chunk_size:
            # Add the page text to the chunk
            current_chunk += page_text
            # If it's the last page, close off the current chunk
            if i == len(pages_text) - 1:
                chunks.append(current_chunk)
                page_ranges.append((chunk_start_page, i))
        else:
            # If the current chunk is not empty, save it before starting a new one
            if current_chunk:
                chunks.append(current_chunk)
                page_ranges.append((chunk_start_page, i - 1))
            # Start a new chunk with the current page
            # But if the page itself exceeds the chunk size, handle it specially
            if len(page_text) > chunk_size:
                # Break down the large page into subchunks, if necessary
                for start in range(0, len(page_text), chunk_size):
                    end = start + chunk_size
                    chunks.append(page_text[start:end])
                    page_ranges.append((i, i))  # This subchunk still corresponds to the current page
                current_chunk = ''  # Reset for the next chunk after handling large page
                chunk_start_page = i + 1  # Next chunk will start from the following page
            else:
                # Otherwise, just start the new chunk normally
                current_chunk = page_text
                chunk_start_page = i
            # Handle case where this is the last page and its text hasn't been added to chunks
            if i == len(pages_text) - 1 and len(page_text) <= chunk_size:
                chunks.append(current_chunk)
                page_ranges.append((chunk_start_page, i))

    return chunks, page_ranges



# Function to create the dataset
def create_dataset(book_paths, chunk_size=1000,num_paginas = 100):
    rows = []
    
    for book_path in book_paths:
        print(f"Processing {book_path}")
        pages_text = extract_text_from_pdf(book_path,num_paginas)
        chunks, page_ranges = split_into_chunks(pages_text, chunk_size)
        
        for chunk, (start_page, end_page) in zip(chunks, page_ranges):
            embedding = embed_with_openai(chunk)
            rows.append({
                'document_name': os.path.basename(book_path),
                'chunk_text': chunk,
                'init_page': start_page + 1,  # Adding 1 to make page numbers human-readable
                'end_page': end_page + 1,    # Adding 1 to make page numbers human-readable
                'embedding': embedding
            })
    
    return pd.DataFrame(rows)


import re
#import unidecode
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')  # Tokenization models

def preprocess_text(text, language='portuguese'):
    # Convert text to lowercase


    text = text.lower()
    
    # Optionally remove accented characters
    #text = unidecode.unidecode(text)
    
    # Tokenize the text using NLTK
    tokens = word_tokenize(text, language=language)
    
    # Optional: Remove punctuation and numbers from tokens
    # You can customize this regular expression based on your needs
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]  # Removes punctuation
 
    
    # Remove tokens that have become empty strings after removal
    tokens = [token for token in tokens if token.strip()]
    
    # Rejoin tokens into a single string separated by space
    clean_text = ' '.join(tokens)
    
    return clean_text