from langchain_community.callbacks.manager import get_openai_callback
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from GroundedAI.chains.chains import ChainFactory
from GroundedAI.callbacks.callbacks import LLMAppCallbackHandler
from GroundedAI.llms.llms import LLMFactory
from GroundedAI.embeddings.embeddings import EmbeddingFactory
from GroundedAI.prompt_templates.rag_prompt_templates import prompt_dict
from GroundedAI.vector_stores.base import VectorStoreFactory
from GroundedAI.retrievers.base import RetrieverFactory

from GroundedAI.utils.config_utils import load_config
from GroundedAI.utils.word_processing import pretty_print_docs, get_contexts_and_metadata

def initialize_qa_retrieval_chain(config_rag):

    # Initializing embedding model
    embedding_model = EmbeddingFactory(provider=config_rag["llm_provider"]).create_model()

    # Initializing LLM
    llm = LLMFactory(provider=config_rag["llm_provider"]).create_model()

    # Load knowledge database
    knowledge_dbs = VectorStoreFactory(
        embedding_model=embedding_model,
        providers=config_rag["vdb_provider"], 
        deployments=config_rag["vdb_deployment"],
        index_names=config_rag["vdb_deployment_index"],
        hybrid=config_rag["hybrid_search"],
        rank_window_size=config_rag["hybrid_search_rank_window"]
        ).create_vector_store()

    retriever = RetrieverFactory(knowledge_dbs,
                                config_rag["search_type"], 
                                config_rag["search_kwargs"], 
                                config_rag["multi_vdb_retriever"], 
                                config_rag["ensemble_retriever_weight"]
                                ).create_retriever()
    
    #Rerank model setup
    rerank_model = HuggingFaceCrossEncoder(model_name=config_rag["rerank_model"]) if config_rag["rerank"] else None
    compressor = CrossEncoderReranker(model=rerank_model, top_n=config_rag["rerank_top_n"]) if config_rag["rerank"] else None

    # Set up QA chain
    qa_chain = ChainFactory(chain_type=config_rag["chain_type"],
                            prompt_template=prompt_dict[config_rag["prompt_template"]],
                            llm=llm,
                            retriever=retriever,
                            compressor=compressor,
                            #callbacks=[MVPMatGPTCallbackHandler()]
                            ).create_chain()

    print("The following is the chain:\n")
    print(qa_chain)

    return qa_chain


async def start_prompting(qa_chain, prompts, batch_mode=True):
    # Start input the prompts and output structural answers, contexts and metata as lists.
    # If not batch_mode, prompts should be a dictionary with {'question': question, 'chat_history': chat_history}
    # If batch_mode, prompts should be a list of dictionaries

    if not batch_mode:
        with get_openai_callback() as cb:
            # One can only use openai callback for non-batch mode
            response = await qa_chain.ainvoke(prompts)
            print(f"   Total Cost (USD): ${cb.total_cost}")
            answer = response["answer"]
            contexts, meta_data = get_contexts_and_metadata(response["source_documents"])

        pretty_print_docs(response["source_documents"])
        return answer, contexts, meta_data
        
    else:
        responses = await qa_chain.abatch(prompts)
        answers, contexts, meta_data = [], [], []
        for response in responses:
            answer = response["answer"]
            contexts_per_question, meta_data_per_question = get_contexts_and_metadata(response["source_documents"])
            answers.append(answer)
            contexts.append(contexts_per_question)
            meta_data.append(meta_data_per_question)
        return answers, contexts, meta_data


async def main():
    config = load_config()

    qa_chain = initialize_qa_retrieval_chain(config["rag_parameters"])

    prompt = "What is a Higgs Boson?"
    chat_history = []
    prompt = {'question': prompt, 'chat_history': chat_history}

    #print(prompt)
    answer, contexts, meta_data= await start_prompting(qa_chain, prompt, batch_mode=False)
    #pretty_print_docs(contexts)
    print(answer)

    #prompts_batch = [{'question': prompt,'chat_history': chat_history}, {'question':prompt_2, 'chat_history': chat_history}]
    #print(prompts_batch)
    #prompts_batch = [prompt, prompt_2]

    #answers, contexts, meta_data = await start_prompting(qa_chain, prompts_batch)

    #print(contexts)

    #print(len(contexts))
    #print(answers)
    #print(meta_data)
    #print("\n")
    #for question_contexts in contexts:
    #    for context in question_contexts:
    #        print(context)
    #        print("______________\n")



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())