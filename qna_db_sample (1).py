# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from app.utils.constant import PREDICT_ROW_COLUMN_STORE,PREDICT_IMAGE_STORE,PREDICT_WORD_STORE
import os,re,pandas as pd
import openai
# Set up your OpenAI API key
openai.api_key = ''

# %%
# predict_fld_path='pdf_ouptut/sample_pg2/predict_table_output'
predict_fld_path="pdf_ouptut/Annual_Report___2022_23__2__bWICfx_87/predict_table_output"
extracted_table_path = predict_fld_path.replace(PREDICT_IMAGE_STORE,PREDICT_ROW_COLUMN_STORE)
extracted_table_path

# %%
all_data = []
for fld_table in sorted(list(filter(lambda x: os.path.isdir(os.path.join(extracted_table_path,x)),os.listdir(extracted_table_path)))):
    for sub_fld in sorted(list(filter(lambda x: os.path.isdir(os.path.join(extracted_table_path,fld_table,x)),os.listdir(os.path.join(extracted_table_path,fld_table))))):
        print(sub_fld)
        name = list(filter(lambda x : re.search('\.csv',x,re.I),os.listdir(os.path.join(extracted_table_path,fld_table,sub_fld))))[0]
        all_data.append(pd.read_csv(os.path.join(extracted_table_path,fld_table,sub_fld,name)))

# %%
# all_data[0]

# %%
# Load pre-trained tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# %%
# Function to generate embeddings and convert to list
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()  # Convert numpy array to list

# %%
# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("financial_data")

# %%
for file,df in enumerate(all_data):
    # df = pd.read_csv(file)
    for i, row in df.iterrows():
        # Combine row data into a single string
        text = " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        embedding = generate_embedding(text)
        # Add data to ChromaDB
        collection.add(
            ids=[f"{file}_row_{i}"],
            embeddings=[embedding],  # Embedding as list
            metadatas=[{"text": text}]
        )

# %%
def retrieve_relevant_data_chroma(question, collection, top_k=5):
    question_embedding = generate_embedding(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    # Assuming results is a dictionary with a list of documents
    relevant_texts = []
    for result in results['metadatas']:
        for doc in result:
            relevant_texts.append(doc['text'])
    return relevant_texts
    # return [result['metadata']['text'] for result in results['documents']]

# %%
def generate_answer_with_openai(question, retrieved_data):
    # Combine retrieved data into a single context string
    context = "\n".join(retrieved_data)

    # Query the GPT-4 model using the ChatCompletion interface
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with expertise in financial data."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=100,
        temperature=0
    )

    return response['choices'][0]['message']['content'].strip()

# %%
# Example usage
# question = "What is EBITDA for 2023?"
# question = "comparision of EBITA for finanicial year 2022 ans 2023 "
# question = "What is sales of Beauty Care ?"
question = "can you please compare sales of Beauty Care with Home care ?"
retrieved_data = retrieve_relevant_data_chroma(question, collection)

# print("Retrieved data:")
# print(retrieved_data)

# Generate the final answer
answer = generate_answer_with_openai(question, retrieved_data)

print("Answer:")
print(answer)

# %%



