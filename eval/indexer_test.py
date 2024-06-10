
import requests
import json
import time
import datetime
from termcolor import colored
from utils.plot_utils import print_results


def run_fetch(kvalue,hash,pergunta):
    kvalue = 3
    fetch = requests.get(f'{url}/api/v2/index/{hash}/fetch?question={pergunta}&kValue={kvalue}', headers=headers)
    dict = json.loads(fetch.text)
    documents = []
    for item in dict:
        print(" - - - - - - - -  - - - - - - - - - - - - - -- - Document - - - - - - - -  - - - - - - - - - - - - - - - - ")
        print(item['text'])
        documents.append(item['text'])
        print(" - - - - - - - -  - - - - - - - - - - - - - -- - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - ")
    
    print("Fetch done")
    return documents

def run_question(pergunta,kvalue=3,model = "gpt-3.5-turbo",temperature = 0.0):
    pergunta = {
            "question":pergunta, 
            "model":model,
            "temperature": temperature,
            "kValue":kvalue
           }
    query = requests.post(f'{url}/api/v3/index/{hash}/customquery', json=pergunta, headers=headers)
    return query
def run_qa_dataset(filename):
    
    with open(filename, 'r') as file:
        data = json.load(file)

    results = []
    for item in data:
        now = datetime.datetime.now()
        pergunta = item['question']
        df_answer = item['answers']['text']
        # Fetch documents and run the question
        # documents = run_fetch(3, hash, pergunta)
        query = run_question(pergunta)
        #print(query)
        
        response_data = query.text
        response_data = json.loads(response_data)
        # Extract the response and second prompt_tokens
        response = response_data.get('response')
        prompt_tokens_list = [usage.get('prompt_tokens') for usage in response_data.get('usage', [])]
        second_prompt_tokens = prompt_tokens_list[1] if len(prompt_tokens_list) > 1 else None
        end_time = datetime.datetime.now()
        
        time_diff = end_time - now
        time_diff = time_diff.total_seconds()
        #color the output

        print_results(pergunta, df_answer, response, second_prompt_tokens, time_diff)
        

        # Add the results to the list
        results.append({
            "question": pergunta,
            "df_answer": df_answer,
            "llm_answer": response,
            "total_time": time_diff,
            "tokens": second_prompt_tokens
        })

    # Create a new JSON file with the results
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_filename = f"results/qa_indexer.json"
    with open(output_filename, 'w') as output_file:
        json.dump(results, output_file, indent=5)

    print(f"Results saved to {output_filename}")
        
    



if __name__ == "__main__":
    import json
    api_key='Cj7MqcmauR-8KDdLoEmRzYuzRKxLVtFIPw4ianmcsFU='
    hash='H94fe3e8d445145dda686b7e8e92e4ea1'
    url ="https://llmindexer-api-dev.saiapplications.com"
    headers = {
    "ApiKey": api_key,
    "Content-Type": "application/json"
    }
    #pergunta = "Can you describe Elena Martinez's professional background, her contributions to environmental policy, her publications, and her philanthropic efforts?"
    #documents = run_fetch(3,hash,pergunta)
    run_qa_dataset("datasets/qa-papers.json")

    
    

    