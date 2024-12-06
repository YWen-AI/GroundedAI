import os

from langchain_elasticsearch import ElasticsearchStore

from elasticsearch import Elasticsearch


def load_ElasticSearch_vectorstore(embedding_model, index, is_hybrid=False, rank_window_size=10):

    es_connection = Elasticsearch(
        cloud_id=os.environ.get('ELASTIC_CLOUD_ID'),
        api_key=os.environ.get('ELASTIC_API_KEY'),
        # api_key = (os.environ.get('ELASTIC_KEY_ID'), os.environ.get('ELASTIC_API_KEY')),
        timeout=30)

    vectorstore = ElasticsearchStore(
    embedding=embedding_model,
    index_name=index,
    es_connection=es_connection,
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=is_hybrid, rrf={"window_size": rank_window_size}),
    )

    return vectorstore
