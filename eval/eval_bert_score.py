import pandas as pd
from bert_score import score
from sentence_transformers import SentenceTransformer, util

class EvaluationMetrics:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def calculate_similarity(self, answer1, answer2):
        embeddings1 = self.model.encode(answer1, convert_to_tensor=True)
        embeddings2 = self.model.encode(answer2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarity.item()
    
    def is_similar(self, similarity_value):
        return 1 if similarity_value >= self.similarity_threshold else 0

    def evaluate(self, llm_answers, df_answers):
        precisions, recalls, f1_scores, similarity_values, similarity_flags = [], [], [], [], []
        for llm_answer, df_answer in zip(llm_answers, df_answers):
            P, R, F1 = score(cands=[llm_answer], refs=[df_answer], lang="pt", rescale_with_baseline=True)
            precisions.append(P.mean().item())
            recalls.append(R.mean().item())
            f1_scores.append(F1.mean().item())
            
            similarity_value = self.calculate_similarity(llm_answer, df_answer)
            similarity_values.append(similarity_value)
            similarity_flags.append(self.is_similar(similarity_value))
            print('------------------                   ------------------                  ------------------')
            print(f"LLM Answer: {llm_answer}\nDF Answer: {df_answer}")
            print(f"Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}, Similarity: {similarity_value}, Is similar: {self.is_similar(similarity_value)}")
            print('------------------                   ------------------                  ------------------')
        return precisions, recalls, f1_scores, similarity_values, similarity_flags

def run_bert_score_evaluation(df_name, similarity_threshold):
    # Ajuste para carregar o DataFrame corretamente
    df_path = df_name + ".pkl"
    df = pd.read_pickle(df_path)
    df.dropna(subset=['df_answer'], inplace=True)
    df.replace({'df_answer': {'N/A': pd.NA}}, inplace=True)
    df.dropna(subset=['df_answer'], inplace=True)
    
    eval_metrics = EvaluationMetrics(similarity_threshold=similarity_threshold)
    
    precisions, recalls, f1_scores, similarity_values, is_similar = eval_metrics.evaluate(df['llm_answer'], df['df_answer'])
    
    # Adicionar os resultados ao dataframe
    df['BS_precision'] = precisions
    df['BS_recall'] = recalls
    df['BS_F1'] = f1_scores
    df['similarity_value'] = similarity_values
    df['is_similar'] = is_similar
    
    # Correção na formação do nome do arquivo CSV para garantir um nome de arquivo válido
    csv_file_path = df_name + "_evaluated.csv"
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    import pandas as pd
    run_bert_score_evaluation(df_name='./__KDTREE__HDBSCAN_____datetime_20240417_103658_____threshold_0.3_____npca_500_____k_3_____execution_time_12492.50', similarity_threshold=0.2)
    #run_bert_score_evaluation(df_name='./__KDTREE__HDBSCAN_____datetime_20240412_184709_____threshold_0.3_____npca_500_____k_3_____execution_time_8863.58', similarity_threshold=0.2)
