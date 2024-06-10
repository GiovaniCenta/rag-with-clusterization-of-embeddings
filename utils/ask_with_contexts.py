import pandas as pd
import time
from utils.log_utils import LogUtils
from utils.openai_utils import ask_with_ai_gateway, ask_with_openai, better_answer
from utils.query_utils import create_query_embedding, create_query_pca,create_query_embedding_ai_gateway

class AskWithContexts:
    def __init__(self, query_text, best_contexts, indexes_context, llm_token_limit,show_results=False):
        self.query_text = query_text
        self.best_contexts = best_contexts
        self.indexes_context = indexes_context
        self.token_limit = llm_token_limit
        self.show_results = show_results
        self.log_utils = LogUtils()

    def ask_with_filtered_contexts(self):
        current_token_count = 0
        best_contexts_text = []
        results_list = []
        results_similarities = []
        st_time = time.time()

        for best_context in self.best_contexts:
            text = best_context[self.indexes_context['chunk_text']]
            num_tokens = len(text)

            if current_token_count + num_tokens > self.token_limit:
                print(f"Token limit reached: {current_token_count} tokens.")
                break

            best_contexts_text.append(text)
            current_token_count += num_tokens

            #llm_answer = ask_with_ai_gateway(query_text, best_contexts_text)
            llm_answer = ask_with_openai(self.query_text, best_contexts_text)
            #llm_answer = better_answer(llm_answer, query_text, best_contexts_text)

            end_time = time.time()
            total_time = end_time - st_time

            results = {
                'question': self.query_text,
                'df_answer': None,  # Placeholder for df_answer, replace with actual df_answer if available
                'llm_answer': llm_answer,
                'total_time': total_time,
                'tokens': current_token_count
            }

            results_log_sim = {
                'question': self.query_text,
                'df_answer': None,  # Placeholder for df_answer, replace with actual df_answer if available
                'similarities_vector': None  # Placeholder for best_similarities, replace with actual best_similarities if available
            }
            results_similarities.append(results_log_sim)
            
        if self.show_results:
            self.log_utils.print_results(self.query_text, None, llm_answer, current_token_count,total_time)
        results_list.append(results)

        results_df = pd.DataFrame(results_list)
        return results_df
