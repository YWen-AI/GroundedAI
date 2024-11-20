from langchain.retrievers import EnsembleRetriever


def create_ensemble_retriever(retrievers, weights):
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)

    return ensemble_retriever


if __name__ == "__main__":
    from GroundedAI.vector_stores.base import VectorStoreFactory

    from GroundedAI.embeddings.embeddings import EmbeddingFactory

    embedding_model = EmbeddingFactory(provider="AzureOpenAI").create_model()

     # Load knowledge database
    knowledge_db_1 = VectorStoreFactory(
        embedding_model=embedding_model, provider="ElasticSearch", domain="Mg").create_vector_store()

    ElasticSearch_retriever_1 = knowledge_db_1.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    knowledge_db_2 = VectorStoreFactory(
        embedding_model=embedding_model, provider="ElasticSearch", domain="Al").create_vector_store()

    ElasticSearch_retriever_2 = knowledge_db_2.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[ElasticSearch_retriever_1, ElasticSearch_retriever_2], weights=[0.5, 0.5]
    )
    ensemble_retriever = create_ensemble_retriever(retrievers=[ElasticSearch_retriever_1, ElasticSearch_retriever_2], weights=[0.5, 0.5])

    docs = ensemble_retriever.invoke("What is corrosion?")
    print(len(docs))
    print(docs)
