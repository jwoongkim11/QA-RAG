from langchain_community.vectorstores import FAISS

class VectorDB:
    def __init__(self, directory, embeddings_model):
        self.directory = directory
        self.embeddings_model = embeddings_model
        self.db = self.load_vector_db() #usage: instance.db

    def load_vector_db(self):
        return FAISS.load_local(self.directory, embeddings=self.embeddings_model)

    def search(self, query, top_k):
        return self.db.similarity_search(query, k=top_k)
