from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import BasePromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch

from GroundedAI.prompt_templates.contextualize_question_prompt_templates import contextualize_q_prompt
from GroundedAI.utils.word_processing import format_docs
from GroundedAI.utils.citation import extract_and_replace_references, format_docs_with_title

def create_conversational_retrieval_chain(llm: BaseLanguageModel, retriever: BaseRetriever, callbacks: BaseCallbackHandler, combine_documents_prompt):

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        callbacks=callbacks,
        verbose=False,
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": combine_documents_prompt}
    )
    
    return qa_chain


def create_conversational_retrieval_chain_LCEL(llm: LanguageModelLike, retriever: RetrieverLike, callbacks: BaseCallbackHandler, combine_documents_prompt: BasePromptTemplate):

    if "question" not in combine_documents_prompt.input_variables:
        raise ValueError(
            "Expected `question` to be a prompt variable, "
            f"but got {combine_documents_prompt.input_variables}"
        )

    history_aware_retriever: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["question"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        contextualize_q_prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="history_aware_chain")

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["source_documents"]))).with_config(run_name="format_docs")
        | combine_documents_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")

    rag_chain_with_source = RunnableParallel(
        {"source_documents": history_aware_retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs).with_config(callbacks=callbacks)
    
    return rag_chain_with_source


def create_conversational_retrieval_chain_LCEL_CrossEncoder_rerank(llm: LanguageModelLike,
                                                    retriever: RetrieverLike,
                                                    callbacks: BaseCallbackHandler,
                                                    combine_documents_prompt: BasePromptTemplate,
                                                    compressor: BaseDocumentCompressor
                                                    ):
    
    if "question" not in combine_documents_prompt.input_variables:
        raise ValueError(
            "Expected `question` to be a prompt variable, "
            f"but got {combine_documents_prompt.input_variables}"
        )
    
    compression_retriever: RetrieverOutputLike = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    
    history_aware_retriever: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["question"]) | compression_retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        contextualize_q_prompt | llm | StrOutputParser() | compression_retriever,
    ).with_config(run_name="history_aware_chain")


    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["source_documents"]))).with_config(run_name="format_docs")
        | combine_documents_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")

    rag_chain_rerank_with_source_rerank = RunnableParallel(
        {"source_documents": history_aware_retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs).with_config(callbacks=callbacks)
    
    return rag_chain_rerank_with_source_rerank


def create_conversational_retrieval_chain_with_citations_LCEL(*, llm: LanguageModelLike, retriever: RetrieverLike, callbacks: BaseCallbackHandler, combine_documents_prompt: BasePromptTemplate):
  
    if "question" not in combine_documents_prompt.input_variables:
        raise ValueError(
            "Expected `question` to be a prompt variable, "
            f"but got {combine_documents_prompt.input_variables}"
        )
        
    history_aware_retriever: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["question"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        contextualize_q_prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="history_aware_chain")

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_title(x["source_documents"]))).with_config(run_name="format_docs")
        | combine_documents_prompt
        | llm
        | StrOutputParser()
        | extract_and_replace_references
    ).with_config(run_name="stuff_documents_chain")

    rag_chain_with_source = RunnableParallel(
        {"source_documents": history_aware_retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs).with_config(callbacks=callbacks)
        
        
    return rag_chain_with_source