from openai import OpenAI
import os
import dotenv
import os
from openai import AzureOpenAI

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def summarize(text):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Resuma o trecho a seguir mantendo as informações principais:"},
        {"role": "user", "content": text},
    ]
    )
    try:
        print(response.choices[0].message.content)

        return response.choices[0].message.content

    except IndexError:
        return 


from openai import OpenAI
import os
import dotenv

def embed_with_openai(text):
    
    response = client.embeddings.create(
    model="text-embedding-3-large",
    input=text
    )
    return response.data[0].embedding


import openai

def ask_with_openai(question="is this a test?", context=["this is not a test", "this is a test, or not?"]):
    # Formatar o contexto para incluir índices e uma separação mais clara

    if len(context) > 1:
        formatted_context = "\n\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(context))
        context = formatted_context
    
    limit_tokens = 16200
    if len(context) > limit_tokens:
        context = context[:limit_tokens]

    message = f"Answer the question: {question}, based solely on the contexts provided below. If you do not find an answer in any of the contexts, report 'N/A':\n\n{context}\n\nNOTE: PLEASE RESPOND AS BRIEFLY AS POSSIBLE."

    #system_message = "You are an assistant who knows nothing. You only use what has been provided to you as context and respond from it with a short and direct answer, including the index of the context used. If not found in the context, return 'N/A'."
    system_message = "You are an assistant who knows nothing. You only use what has been provided to you as context and respond from it with a short and direct answer. If not found in the context, return 'N/A'."

    system_message = ("Context information is below. \n"
    "---------------------\n"
    "{context}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}.\n" 
    "If the context doesn't contain the information needed, answer \"I dont know\".\n")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":f"{system_message}" },
            {"role": "user", "content": f"{message}"},
        ]
    )
    
    return completion.choices[0].message.content


import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


def ask_with_ai_gateway(question,context):

    client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    message = f"Answer the question: {question}, based solely on the contexts provided below. If you do not find an answer in any of the contexts, report 'N/A':\n\n{context}\n\nNOTE: PLEASE RESPOND AS BRIEFLY AS POSSIBLE."

    #system_message = "You are an assistant who knows nothing. You only use what has been provided to you as context and respond from it with a short and direct answer, including the index of the context used. If not found in the context, return 'N/A'."
    system_message = "You are an assistant who knows nothing. You only use what has been provided to you as context and respond from it with a short and direct answer. If not found in the context, return 'N/A'."

    system_message = ("Context information is below. \n"
    "---------------------\n"
    "{context}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}.\n" 
    "If the context doesn't contain the information needed, answer \"I dont know\".\n")
    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo-1106", # model = "deployment_name".
            messages=[
                {"role": "system", "content": f"{system_message}"},
                {"role": "user", "content": f"{message}"}
            ]
        )
        return response.choices[0].message.content
    except openai.BadRequestError as e:
        print(f"Encountered a safety filter issue: {e}")
        return "SAFETY ISSUE"
    except openai.NotFoundError as e:
        print(f"Encountered issue: {e}")
        return "DEPLOYMENT NOT FOUND ISSUE"
    
    
import os
from openai import AzureOpenAI
def embbed_with_ai_gateway(text):
    endpoint= os.getenv("AZURE_OPENAI_ENDPOINT")
    secretValue= os.getenv("AZURE_OPENAI_API_KEY")
    client = AzureOpenAI(
    api_key = secretValue,  
    api_version = "2024-02-01",
    azure_endpoint =endpoint 
    )

    response = client.embeddings.create(
        input = text,
        model= "text-embedding-3-large"
    )

    return response.data[0].embedding
    
    
import numpy as np

def should_ask_again(previous_answer, unsatisfactory_responses=["n/a","DEPLOYMENT NOT FOUND ISSUE", "not found in this context", "information not found", "no relevant data", "I don't know.","I dont know","I dont know."]):

    # Normalize the previous answer to lower case for comparison
    normalized_answer = previous_answer.lower().strip()

    # Check if any unsatisfactory response is a substring of the normalized answer
    for phrase in unsatisfactory_responses:
        if phrase in normalized_answer:
            return True

    return False

def better_answer(previous_answer, question, context):
    trys = 3
    answer = previous_answer 
    while should_ask_again(answer):
        print(f"Try = {trys}  - Re-querying AI due to unsatisfactory previous answer.")
        trys -= 1
        answer = ask_with_ai_gateway(question, context)
        if trys == 0:
            break
    return answer

# Lembre-se de configurar sua API key do OpenAI
if __name__ == "__main__":
    import pandas as pd
    dotenv.load_dotenv()
    print(embbed_with_ai_gateway("This is a test"))
    print(ask_with_ai_gateway("Who were the founders of Microsoft?","Bill Gates and Paul Allen"))
    file_name = "sec-10-small_chunks1000t.pkl"
    df_small = pd.read_pickle(file_name)
    df_small['embedding'] = df_small['chunk_text'].apply(lambda x: embbed_with_ai_gateway(x))
    df_small.to_pickle("sec-10-small_chunks1000t_embedded.pkl")
