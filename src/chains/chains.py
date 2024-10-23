from src.chains.conversational_retrieval_chain import (
    create_conversational_retrieval_chain, 
    create_conversational_retrieval_chain_LCEL,
    create_conversational_retrieval_chain_LCEL_CrossEncoder_rerank,
    create_conversational_retrieval_chain_with_citations_LCEL
    )

class ChainFactory():
    def __init__(self, chain_type, prompt_template, llm, retriever=None, output_parser=None, callbacks=None, compressor=None):
        self.chain_type = chain_type
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.callbacks = callbacks
        self.output_parser = output_parser
        self.compressor = compressor

    def create_chain(self):
        if self.chain_type == "conversational_retrieval_chain":
            return create_conversational_retrieval_chain(
                llm = self.llm,
                retriever = self.retriever,
                callbacks = self.callbacks,
                combine_documents_prompt = self.prompt_template
            )
        
        elif self.chain_type == "conversational_retrieval_chain_LCEL":
            return create_conversational_retrieval_chain_LCEL(
                llm = self.llm,
                retriever = self.retriever,
                callbacks = self.callbacks,
                combine_documents_prompt = self.prompt_template
            )
        
        elif self.chain_type == "conversational_retrieval_chain_LCEL_CrossEncoder_rerank":
            return create_conversational_retrieval_chain_LCEL_CrossEncoder_rerank(
                llm = self.llm,
                retriever = self.retriever,
                callbacks = self.callbacks,
                combine_documents_prompt = self.prompt_template,
                compressor = self.compressor
            )
        
        elif self.chain_type == "conversational_retrieval_chain_with_citations_LCEL":
            return create_conversational_retrieval_chain_with_citations_LCEL(
                llm = self.llm,
                retriever = self.retriever,
                callbacks = self.callbacks,
                combine_documents_prompt = self.prompt_template
            )

        elif self.chain_type == "simple_chat_llm_chain":
            chain = self.prompt_template | self.llm | self.output_parser
            return chain
        
        else:
            raise ValueError("Invalid chain type. Expected 'conversational_retrieval_chain'.")