import faiss
import numpy as np
import pickle
from typing import List
import os

class RAGRetriever:
    def __init__(self):
        self.index = None
        self.embeddings = None
        self.documents = None
        self._load_data()
    
    def _load_data(self):
        """Load embeddings and documents from pickle file"""
        try:
            # Load embeddings and documents
            with open('Embeddings.pkl', 'rb') as f:
                data = pickle.load(f)
                # Convert embeddings to numpy array if they aren't already
                self.embeddings = np.array(data['embeddings'], dtype=np.float32)
                self.documents = data['documents']
            
            # Create and populate FAISS index
            dimension = self.embeddings.shape[1]  # Get dimension from embeddings
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            
            print(f"Successfully loaded {len(self.documents)} documents and created index")
            print(f"Embeddings shape: {self.embeddings.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings for a text using OpenAI's API"""
        try:
            import openai
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response['data'][0]['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise

    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents"""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get the relevant documents
            relevant_docs = [self.documents[i] for i in indices[0]]
            
            return relevant_docs
            
        except Exception as e:
            print(f"Search error: {e}")
            return ["Error retrieving relevant information."]

# Global retriever instance
_retriever = None

def get_retriever():
    """Get or create the global retriever instance"""
    global _retriever
    if _retriever is None:
        try:
            _retriever = RAGRetriever()
        except Exception as e:
            print(f"Failed to initialize retriever: {e}")
    return _retriever

def search_rag(query: str) -> str:
    """Search function to be used by the main application"""
    retriever = get_retriever()
    if retriever is None:
        return "Retriever not initialized. Please check the index and embeddings files."
    
    try:
        relevant_docs = retriever.search(query)
        return " ".join(relevant_docs) if relevant_docs else "No relevant information found."
    except Exception as e:
        print(f"Search error: {e}")
        return f"Error during search: {str(e)}"