from GroundedAI.vector_stores.faiss import load_FAISS_vectorstore
from GroundedAI.vector_stores.elastic_search import load_ElasticSearch_vectorstore

from GroundedAI.utils.config_utils import load_config
from GroundedAI.utils.load_env_keys import load_keys


class VectorStoreFactory():
    def __init__(self, embedding_model, providers, deployments, index_names, hybrid=False, rank_window_size=10):

        config = load_config()["vector_db_parameters"]

        self.embedding_model = embedding_model
        self.providers = providers if isinstance(providers, list) else [providers]
        self.deployments = deployments if isinstance(deployments, list) else [deployments]
        self.index_names = index_names if isinstance(index_names, list) else [index_names]
        self.index_values = config["ElasticSearch"]["indexes"] if "ElasticSearch" in self.providers else None
        self.is_hybrid = hybrid if "ElasticSearch" in self.providers else False
        self.rank_window_size = rank_window_size if "ElasticSearch" in self.providers else None
        self.paths = config["FAISS"]["paths"] if "FAISS" in self.providers else None

        if "ElasticSearch" in self.providers:
            load_keys(config["ElasticSearch"]["keys"], deployments)

    def create_vector_store(self) -> list:
        vector_stores = []
        for provider in self.providers:
            if provider == "FAISS":
                for index_name in self.index_names:
                    vector_stores.append(load_FAISS_vectorstore(path=self.paths[index_name], embedding_model=self.embedding_model))
            if provider == "ElasticSearch":
                for index_name in self.index_names:
                    vector_stores.append(
                        load_ElasticSearch_vectorstore(
                            embedding_model=self.embedding_model,
                            index=self.index_values[index_name],
                            is_hybrid=self.is_hybrid,
                            rank_window_size=self.rank_window_size
                            )
                        )
            if provider not in ["FAISS", "ElasticSearch"]:
                raise ValueError("Invalid provider. Expected 'FAISS' or 'ElasticSearch'.")

        return vector_stores


if __name__ == "__main__":
    from src.embeddings.embeddings import EmbeddingFactory
    embedding_model = EmbeddingFactory(provider="AzureOpenAI").create_model()

    # Load knowledge database
    knowledge_db_1 = VectorStoreFactory(
        embedding_model=embedding_model, providers="ElasticSearch", deployments="Mg").create_vector_store()

    print(knowledge_db_1)
