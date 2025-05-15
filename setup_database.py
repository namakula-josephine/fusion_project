from flask import Flask, request, jsonify
import openai
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv('secrets.env')

# Load the OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the embeddings and processed lines
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('processed_lines.pkl', 'rb') as f:
    processed_lines = pickle.load(f)

# Function to retrieve the most relevant text based on a query
def retrieve_relevant_text(query, embeddings, texts, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    return [texts[i] for i in top_indices]


@app.route('/')
def index():
    return "RAG Flask API is running."


# Route to handle incoming queries
@app.route('/query', methods=['GET','POST'])
def query():
    data = request.get_json()
    query = data['query']
    relevant_texts = retrieve_relevant_text(query, embeddings, processed_lines)
    
    response_texts = []
    for text in relevant_texts:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text + "\n\n" + query}
            ],
            max_tokens=150
        )
        response_texts.append(response.choices[0].message['content'].strip())

    return jsonify({'responses': response_texts})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




# from Flask import Flask, requests, render_template, jsonify, session
# from sentence_transformers import SentenceTransformer, util
# import pandas as pd
# from flask_session import Session
# import sqlite3
# import pickle
# import numpy as np
# import faiss
# import openai
# import os
# from dotenv import load_dotenv

# app = Flask(__name__)
# app.secret_key = '3f73c70262a4677b74cff9c79dbd1344'
# app.config['SESSION_TYPE'] = 'filesystem'
# Session(app)

# # Load OpenAI API Key
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # File paths
# DB_NAME = "D:/fusion_project/Potato_blight.db"
# TABLE_NAME = "documents"
# FAISS_INDEX_PATH = "D:/fusion_project/faiss_index.index"
# CORPUS_FILE = "D:/fusion_project/data/myrag.txt"  # Path to your large text file

# def load_corpus_from_file(file_path, chunk_size=1000):
#     """Load documents from a text file in chunks to handle large datasets."""
#     corpus = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             doc = line.strip()
#             if doc:  # Skip empty lines
#                 corpus.append(doc)
#                 if len(corpus) >= chunk_size:
#                     yield corpus
#                     corpus = []
#         if corpus:
#             yield corpus

# def create_document_embeddings(docs):
#     """Generate embeddings for a list of documents using OpenAI's text-embedding-3-large."""
#     embeddings = []
#     for doc in docs:
#         try:
#             response = openai.embeddings.create(
#                 model="text-embedding-3-large",  # Updated to the latest model
#                 input=[doc]
#             )
#             embeddings.append(np.array(response.data[0].embedding, dtype=np.float32))
#         except openai.OpenAIError as e:
#             print(f"🔴 OpenAI Embedding Error for document '{doc}': {e}")
#             embeddings.append(None)
#     return np.array([emb for emb in embeddings if emb is not None], dtype=np.float32)

# def create_database_and_faiss_index(db_name, table_name, corpus_file):
#     # Connect to SQLite database
#     conn = sqlite3.connect(db_name)
#     cursor = conn.cursor()

#     # Create table if it doesn't exist
#     cursor.execute(f"""
#     CREATE TABLE IF NOT EXISTS {table_name} (
#         id INTEGER PRIMARY KEY,
#         text TEXT,
#         vector BLOB
#     )
#     """)

#     # Initialize FAISS index
#     # Note: text-embedding-3-large has a dimension of 3072 (confirm with API response)
#     dimension = 3072  # Default dimension for text-embedding-3-large
#     faiss_index = faiss.IndexFlatL2(dimension)

#     # Process the corpus in chunks
#     for chunk_idx, corpus_chunk in enumerate(load_corpus_from_file(corpus_file)):
#         print(f"Processing chunk {chunk_idx + 1} with {len(corpus_chunk)} documents...")

#         # Generate embeddings for the chunk
#         vectors = create_document_embeddings(corpus_chunk)
#         if len(vectors) == 0:
#             print(f"🔴 No valid embeddings generated for chunk {chunk_idx + 1}. Skipping...")
#             continue

#         # Verify embedding dimension (in case it differs from expected)
#         if vectors.shape[1] != dimension:
#             print(f"🔴 Embedding dimension mismatch in chunk {chunk_idx + 1}. Expected {dimension}, got {vectors.shape[1]}. Adjusting dimension...")
#             dimension = vectors.shape[1]  # Update dimension to match actual embeddings
#             faiss_index = faiss.IndexFlatL2(dimension)  # Recreate index with correct dimension
#             faiss_index.add(vectors)  # Add all previous vectors again (if any)
#         faiss_index.add(vectors)

#         # Insert documents and vectors into the database
#         for doc, vector in zip(corpus_chunk, vectors):
#             vector_blob = pickle.dumps(vector)
#             cursor.execute(f"INSERT INTO {table_name} (text, vector) VALUES (?, ?)", (doc, vector_blob))

#         # Commit after each chunk to save progress
#         conn.commit()

#     # Save FAISS index to disk
#     faiss.write_index(faiss_index, FAISS_INDEX_PATH)
#     print(f"FAISS index saved to {FAISS_INDEX_PATH}")

#     # Final commit and close database connection
#     conn.commit()
#     conn.close()
#     print("Database created and populated successfully.")

# if __name__ == "__main__":
#     create_database_and_faiss_index(DB_NAME, TABLE_NAME, CORPUS_FILE)