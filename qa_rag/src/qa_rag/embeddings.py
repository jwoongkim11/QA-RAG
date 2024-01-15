from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_llm_embedder_model(model_name="BAAI/llm-embedder", device="cuda", normalize_embeddings=True):
    # Set the model parameter
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}

    # Creating embedding model instance
    llm_embedder_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this query for retrieving relevant documents: "
    )
    return llm_embedder_model
