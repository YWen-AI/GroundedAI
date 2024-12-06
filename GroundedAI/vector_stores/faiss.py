from langchain_community.vectorstores import FAISS


def create_FAISS_vectorstore(documents, embedding_model):
    return FAISS.from_documents(documents, embedding_model)


def load_FAISS_vectorstore(path, embedding_model):
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
