import pandas as pd
from bert_score import score
from sentence_transformers import SentenceTransformer, util

class EvaluationMetrics:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def calculate_similarity(self, answer1, answer2):
        # Ensure inputs are strings and not empty
        answer1, answer2 = str(answer1), str(answer2)
        embeddings1 = self.model.encode(answer1, convert_to_tensor=True)
        embeddings2 = self.model.encode(answer2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarity.item()
    
    def is_similar(self, similarity_value):
        return 1 if similarity_value >= self.similarity_threshold else 0

    def evaluate(self, llm_answers, df_answers):
        precisions, recalls, f1_scores, similarity_values, similarity_flags = [], [], [], [], []
        for llm_answer, df_answer in zip(llm_answers, df_answers):
            # Convert to string and handle NaNs or other non-string data
            llm_answer, df_answer = str(llm_answer) if pd.notna(llm_answer) else "", str(df_answer) if pd.notna(df_answer) else ""
            P, R, F1 = score(cands=[llm_answer], refs=[df_answer], lang="pt", rescale_with_baseline=True)
            precisions.append(P.mean().item())
            recalls.append(R.mean().item())
            f1_scores.append(F1.mean().item())
            
            similarity_value = self.calculate_similarity(llm_answer, df_answer)
            similarity_values.append(similarity_value)
            similarity_flags.append(self.is_similar(similarity_value))
            print("answer1:", llm_answer, "answer2:", df_answer, "similarity:", similarity_value, "is_similar:", self.is_similar(similarity_value))            
        return precisions, recalls, f1_scores, similarity_values, similarity_flags

def run_bert_score_evaluation(df_path, similarity_threshold):
    df = pd.read_pickle(df_path)

    # Ensure the expected columns are correctly named
    df.rename(columns={'rag_answer': 'llm_answer', 'answer': 'df_answer'}, inplace=True)
    
    # Drop rows where df_answer is NaN and replace 'N/A' with pd.NA
    df.dropna(subset=['df_answer'], inplace=True)
    df.replace({'df_answer': {'N/A': pd.NA}}, inplace=True)
    df.dropna(subset=['df_answer'], inplace=True)
    
    eval_metrics = EvaluationMetrics(similarity_threshold=similarity_threshold)

    
    df = df.head(200)
    precisions, recalls, f1_scores, similarity_values, is_similar = eval_metrics.evaluate(df['llm_answer'].fillna(''), df['df_answer'].fillna(''))
    
    
    # Add the results to the dataframe
    df['BS_precision'] = precisions
    df['BS_recall'] = recalls
    df['BS_F1'] = f1_scores
    df['similarity_value'] = similarity_values
    df['is_similar'] = is_similar
    
    # Save the enhanced dataframe
    csv_file_path = df_path.replace('.pkl', 'PAPERS_evaluated.csv')
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    import os
    results_folder = './results'
    for file in os.listdir(results_folder):
        if file.endswith('.pkl'):
            df_path = os.path.join(results_folder, file)
            run_bert_score_evaluation(df_path, similarity_threshold=0.2)
