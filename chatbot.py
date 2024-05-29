import os
import pandas as pd
from openai import OpenAI
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


def search(qdrant_client, encoder, database_name, user_prompt):
    
    hits = qdrant_client.search(
    collection_name = database_name,
    query_vector = encoder.encode(user_prompt).tolist(),
    limit=3)

    # define a variable to hold the searching results
    search_results = [hit.payload for hit in hits]

    return search_results

def get_completion_from_messages(client, messages, model, temperature=0):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature # this is the degree of randomness of model
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    # Init
    DATABASE_NAME = 'top_wines'
    MODEL_ENCODE_NAME = 'all-MiniLM-L6-v2'
    DATA_PATH = './top_rated_wines.csv'
    OPENAI_API_KEY_PATH = '/home/loc/Documents/OPENAI_API_KEY.txt'
    MODEL_LLM_NAME = 'gpt-3.5-turbo'
    system_message = 'You are chatbot, a wine specialist, Your top priority is to help the customer select wine'
    user_prompt = ''
    os.environ['TOKENIZERS_PARALLELISM']='False'

    # load OPENAI_API_KEY
    with open(OPENAI_API_KEY_PATH) as f:
        OPENAI_API_KEY = f.read().strip()

    print('OPENAI_API_KEY Loaded!')

    # import data
    df = pd.read_csv(DATA_PATH)
    df = df[df['variety'].notna()]
    data = df.to_dict('records')
    print('Data Imported!')

    # create database vector
    # Load the model to create embeddings
    encoder = SentenceTransformer(MODEL_ENCODE_NAME)

    # create the vector database client
    qdrant = QdrantClient(":memory:") # create in-memory Qdrant instance

    # create vector config
    vector_config = models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance = models.Distance.COSINE
    )

    # create collection to store books
    qdrant.recreate_collection(
        collection_name = DATABASE_NAME,
        vectors_config = vector_config
    )

    # vectorize!
    qdrant.upsert(
        collection_name = DATABASE_NAME,
        points = [ models.PointStruct(id = idx,
                                    vector = encoder.encode(doc['notes']).tolist(), 
                                    payload = doc
                                    ) 
                for idx, doc in enumerate(data) ]
    )

    print("VectorDatabase Created!")

    client = OpenAI(
    api_key = OPENAI_API_KEY
    )
    print("OpenAI client Created!")

    context = [{'role':'system','content':system_message}]
    print('Context Created!')
    
    while True:

        # create instance of client openai
        user_prompt = input("Enter your request:\n")
        #print(user_prompt)

        if user_prompt == 'exit':
            break

        context = [{'role':'system','content':system_message},
        {'role':'user','content':user_prompt}]

        search_result = search(qdrant, encoder, DATABASE_NAME, user_prompt)
        #print(search_result)

        response = get_completion_from_messages(client, context, MODEL_LLM_NAME, temperature=0)

        print("chatGPT:\n",response,'\n')

    print('Bye Bye!')
    exit(0)