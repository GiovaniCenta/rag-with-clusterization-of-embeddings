from openai import OpenAI
import os
import dotenv
client = OpenAI(api_key=('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'))
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
    model="text-embedding-3-small",
    input=text
    )
    return response.data[0].embedding


import openai

def ask_with_openai(question="is this a test?", context=["this is not a test", "this is a test, or not?"]):
    # Formatar o contexto para incluir índices e uma separação mais clara

    if len(context) > 1:
        formatted_context = "\n\n".join(f"Contexto {i+1}: {ctx}" for i, ctx in enumerate(context))
        context = formatted_context

    #message = f"Responda a pergunta: {question}, com base apenas nos contextos a seguir. Indique o número do contexto utilizado na sua resposta. Caso não encontre em nenhum dos contextos, informe que não encontrou com 'N/A':\n\n{context}\n\nOBSERVAÇÃO: RESPONDA DA FORMA MAIS BREVE POSSÍVEL."
    message = f"Responda a pergunta: {question}, com base apenas nos contextos a seguir. Caso não encontre em nenhum dos contextos, informe que não encontrou com 'N/A':\n\n{context}\n\nOBSERVAÇÃO: RESPONDA DA FORMA MAIS BREVE POSSÍVEL."
    #system_message = "Você é um assistente que não sabe nada. Você utiliza apenas o que foi fornecido para você como contexto e responde a partir disso com uma resposta curta e direta, incluindo o índice do contexto utilizado. Caso não encontre no contexto, retorne 'N/A'."
    system_message = "Você é um assistente que não sabe nada. Você utiliza apenas o que foi fornecido para você como contexto e responde a partir disso com uma resposta curta e direta,  Caso não encontre no contexto, retorne 'N/A'."

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":f"{system_message}" },
            {"role": "user", "content": f"{message}"},
        ]
    )
    
    return completion.choices[0].message.content

# Lembre-se de configurar sua API key do OpenAI
if __name__ == "__main__":
    

    # Exemplo de uso
    question = "Isso é um teste?"
    context = ["Isso não é um teste", "Isso é um teste, ou não?"]
    print(ask_with_openai(question, context))
