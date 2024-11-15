from langchain.retrievers import MergerRetriever

def create_merger_retriever(retrievers):
    merger_retriever = MergerRetriever(retrievers=retrievers)

    return merger_retriever


if __name__ == "__main__":
    from src.vector_stores.base import VectorStoreFactory
    from src.utils.config_utils import load_config
    from src.embeddings.embeddings import EmbeddingFactory

    config = load_config("src/config.json")

    embedding_model = EmbeddingFactory(provider="AzureOpenAI").create_model()

    # Load knowledge database
    knowledge_db_1 = VectorStoreFactory(
        embedding_model=embedding_model, provider="ElasticSearch", domain="Mg").create_vector_store()

    ElasticSearch_retriever_1 = knowledge_db_1.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    knowledge_db_2 = VectorStoreFactory(
        embedding_model=embedding_model, provider="ElasticSearch", domain="Al").create_vector_store()

    ElasticSearch_retriever_2 = knowledge_db_2.as_retriever(search_type="similarity", search_kwargs={"k": 5})


    merger_retriever = create_merger_retriever([ElasticSearch_retriever_1, ElasticSearch_retriever_2])

    docs = merger_retriever.invoke("What is corrosion?")

    print(len(docs))

    print(docs)
