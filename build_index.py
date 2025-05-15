import faiss
import numpy as np
import openai
import pickle
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../../secrets.env')

def load_documents() -> List[str]:
    """Load documents from your source (modify this based on your data source)"""
    # Example: Loading from a text file
    documents = []
    try:
        with open('potato_disease_data.txt', 'r', encoding='utf-8') as f:
            documents = f.read().split('\n\n')  # Split by double newline
    except Exception as e:
        print(f"Error loading documents: {e}")
    return documents

def create_embeddings(documents: List[str]) -> np.ndarray:
    """Create embeddings for all documents using OpenAI's API"""
    embeddings = []
    
    for doc in documents:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=doc
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            continue
    
    return np.array(embeddings, dtype=np.float32)

def load_existing_embeddings() -> Dict:
    """Load existing embeddings and documents from Embeddings.pkl"""
    try:
        with open('Embeddings.pkl', 'rb') as f:
            stored_data = pickle.load(f)
            print(f"Loaded {len(stored_data['documents'])} documents and their embeddings")
            return stored_data
    except Exception as e:
        print(f"Error loading existing embeddings: {e}")
        return None

def build_index():
    """Build the FAISS index from existing embeddings"""
    try:
        # Load existing embeddings and documents
        stored_data = load_existing_embeddings()
        if not stored_data:
            raise ValueError("No embeddings found in Embeddings.pkl")

        embeddings = np.array(stored_data['embeddings'], dtype=np.float32)
        documents = stored_data['documents']

        # Create and train FAISS index
        dimension = len(embeddings[0])
        print(f"Creating index with dimension {dimension}")
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save the index
        faiss.write_index(index, 'faiss_index.index')
        print(f"Successfully built and saved index with {index.ntotal} vectors")
        
    except Exception as e:
        print(f"Error building index: {e}")

if __name__ == "__main__":
    build_index()