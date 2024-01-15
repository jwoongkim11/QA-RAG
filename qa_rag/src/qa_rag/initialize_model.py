from embeddings import get_llm_embedder_model
from reranker import Reranker
import pickle

def initialize_models():
    # Initialize embedding model
    llm_embedder_model = get_llm_embedder_model()

    # Initialize Reranker
    reranker_model_name = 'BAAI/bge-reranker-large'
    reranker = Reranker(reranker_model_name)

    # Save models to files
    with open('llm_embedder_model.pkl', 'wb') as f:
        pickle.dump(llm_embedder_model, f)
    with open('reranker.pkl', 'wb') as f:
        pickle.dump(reranker, f)

if __name__ == "__main__":
    initialize_models()
    print("Initializing model completed")
