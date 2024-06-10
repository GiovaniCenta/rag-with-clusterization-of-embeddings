import pandas as pd
from termcolor import colored
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class LogUtils:
    @staticmethod
    def print_results(pergunta, df_answer, response, second_prompt_tokens, time_diff):
        header = colored(f"{'Response info':^60}", 'blue', 'on_white', attrs=['bold', 'underline'])  # Header styling
        divider = colored('-' * 60, 'red')  # Divider styling
        pergunta_colored = colored(f"{'Question:':<10}", 'green') + f"{pergunta}"
        df_answer_colored = colored(f"{'Correct Answer:':<10}", 'magenta') + f"{df_answer}"
        response_colored = colored(f"{'Test Answer:':<10}", 'yellow') + f"{response}"
        second_prompt_tokens_colored = colored(f"{'prompt_tokens:':<10}", 'red') + f"{str(second_prompt_tokens)}"
        time_colored = colored(f"{'Time:':<10}", 'cyan') + f"{time_diff:.2f}s"

        print(header)
        print(pergunta_colored)
        print(df_answer_colored)
        print(response_colored)
        print(second_prompt_tokens_colored)
        print(time_colored)
        print(divider)

    @staticmethod
    def show_findings(best_chunks_info, best_similarities, indexes):
        header = colored(f"{'Best Chunk Infos':^60}", 'blue', 'on_white', attrs=['bold', 'underline'])  # Header styling
        divider = colored('-' * 60, 'red')  # Divider styling

        print(header)
        print(divider)

        for index, (chunk_row, similarity) in enumerate(zip(best_chunks_info, best_similarities), start=1):
            clean_chunk_text = chunk_row[indexes['chunk_text']].replace('\t', ' ').replace('\n', ' ')[:150]  # Truncated text for display
            document_name = chunk_row[indexes['document_name']]
            cluster_label = chunk_row[indexes['cluster_label']]

            print(f"Chunk #{index}:")
            similarity_row = colored(f"{'Best Similarity:':<20}", 'yellow') + f"{similarity:.4f}"
            document_row = colored(f"{'Document Name:':<20}", 'green') + document_name
            cluster_row = colored(f"{'Cluster:':<20}", 'blue') + str(cluster_label)
            text_row = colored(f"{'Chunk Text:':<20}", 'cyan') + clean_chunk_text + "..."

            print(similarity_row)
            print(document_row)
            print(cluster_row)
            print(text_row)
            print(divider)  # Print divider after each chunk's information

    @staticmethod
    def plot_clusters(data, indexes):
        pca_components = np.vstack(data[:, indexes["pca_components"]])
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(pca_components)

        plt.figure(figsize=(12, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data[:, indexes['cluster_label']], cmap='viridis', alpha=0.6)
        plt.colorbar()
        plt.title('t-SNE visualization of clustering')
        plt.xlabel('t-SNE component 1')
        plt.ylabel('t-SNE component 2')
        plt.show()

    @staticmethod
    def log_results(output_file, query_text, df_answer, llm_answer, current_token_count, total_time, best_contexts, best_similarities, indexes_context):
        output_file.write(" = = = = = = = = = = = = = = = = = = = = = = == = = = = = = = = = = == = = = = = = = = = = = = = == \n")
        output_file.write(f"Question: {query_text}\n")
        output_file.write(f"Answer: {df_answer}\n")
        output_file.write(f"Answer from LLM: {llm_answer}\n")
        output_file.write(f"Total time: {total_time:.2f}\n")
        output_file.write(f"Tokens: {current_token_count}\n")
        output_file.write("Best contexts:\n")
        for best_context, best_similarity in zip(best_contexts, best_similarities):
            output_file.write(f"Similarity: {best_similarity} || Chunk: {best_context[indexes_context['chunk_text']]}\n")
        output_file.write("Similarties\n")
        output_file.write(f"{best_similarities}\n")
        output_file.write(" = = = = = = = = = = = = = = = = = = = = = = == = = = = = = = = = = = == = = = = = = = = = = = = = == \n")
        output_file.write("\n")

# Example usage:
# LogUtils.print_results(pergunta, df_answer, response, second_prompt_tokens, time_diff)
# LogUtils.show_findings(best_chunks_info, best_similarities, indexes)
# LogUtils.plot_clusters(data, indexes)
# with open('output_log.txt', 'w') as f:
#     LogUtils.log_results(f, query_text, df_answer, llm_answer, current_token_count, total_time, best_contexts, best_similarities, indexes_context)
