from src.retrievers.simple_retriever import create_simple_retriever
from src.retrievers.ensemble_retriever import create_ensemble_retriever
from src.retrievers.merger_retriever import create_merger_retriever

class RetrieverFactory():
    def __init__(self, vector_stores, search_type, search_kwargs, multi_vector_store_strategies=None, weights=None):

        self.vector_stores = vector_stores
        self.multi_vector_store_strategies = multi_vector_store_strategies
        self.weights = weights if self.multi_vector_store_strategies == "ensemble" else None

        self.search_type = search_type
        self.search_kwargs = search_kwargs

    def create_retriever(self):
        for vector_store in self.vector_stores:
            if self.multi_vector_store_strategies == "ensemble":
                retrievers = []
                for vector_store in self.vector_stores:
                    retrievers.append(create_simple_retriever(vector_store, self.search_type, self.search_kwargs))
                return create_ensemble_retriever(retrievers, self.weights)
            
            elif self.multi_vector_store_strategies == "merger":
                retrievers = []
                for vector_store in self.vector_stores:
                    retrievers.append(create_simple_retriever(vector_store, self.search_type, self.search_kwargs))
                return create_merger_retriever(retrievers)
            
            elif self.multi_vector_store_strategies == None:
                return create_simple_retriever(vector_store, self.search_type, self.search_kwargs)
            
            else:
                raise ValueError("Creating retriver fails!")


if __name__ == "__main__":
    from src.vector_stores.base import VectorStoreFactory

    from src.embeddings.embeddings import EmbeddingFactory
    embedding_model = EmbeddingFactory(provider="AzureOpenAI").create_model()

    # Load knowledge databases
    knowledge_dbs = VectorStoreFactory(
        embedding_model=embedding_model, providers="ElasticSearch", deployments=["Al", "Mg"]).create_vector_store()

    print(len(knowledge_dbs))

    retriever = RetrieverFactory(knowledge_dbs, "similarity", {"k": 2}, "ensemble", [0.5, 0.5]).create_retriever()
    
    print(retriever)

    prompt = "We are working with an Aluminum-8.5wt% Magnesium alloy. The alloy exhibits corrosion in the nitric acid mass loss test. What could be the possible reason for this observation?"
    
    docs = retriever.invoke(prompt)

    print(docs)