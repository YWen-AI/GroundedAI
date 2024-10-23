def create_simple_retriever(vector_store, search_type, search_kwargs):
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
