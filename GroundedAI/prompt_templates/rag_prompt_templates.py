from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

# Basic templates
user_template = """
{question}
"""

citation_system_template = """Each SENTENCE in your response should be followed by citations indicating \
the SOURCE TITLE from which the information was taken in [source: Source Title 1] [source: Source Title 2] ... [source: Source Title N] format.
DO NOT add references list at the end of the answer.
"""

####################
# Default QA RAG prompt templates
system_template = """You are an experienced experimental particle scientist.
You are only allowed to answer questions in the field of physics.
If the question is not related to physics, you should reply politely that you can only answer questions in the field of physics.
And you should use only the following pieces of context to help answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}
"""

messages_rag = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template),
]

default_qa_prompt = ChatPromptTemplate.from_messages(messages_rag)

####################
# Default Strawberry QA RAG prompt templates
strawberry_system_template = """You are an experienced experimental particle scientist.
You are only allowed to answer questions in the field of physics.
If the question is not related to physics, you should reply politely that you can only answer questions in the field of physics.
And you should use only the following pieces of context to help answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

{question}
"""

messages_rag = [
    HumanMessagePromptTemplate.from_template(strawberry_system_template),
]

default_strawberry_qa_prompt = ChatPromptTemplate.from_messages(messages_rag)


####################
# Brute-force QA RAG prompt templates
brute_force_system_template = """
You are only allowed to answer questions in the field of physics.
If the question is not related to physics, you should reply politely that you can only answer questions in the field of physics.
You are going to answer the user's question using the contexts below.
Keep your answer ground in the facts of the contexts.
If the contexts don't contain the facts to answer the question, reply politely that the given contexts are not sufficient to answer the question.

{context}
"""

brute_force_messages_rag = [
    SystemMessagePromptTemplate.from_template(brute_force_system_template),
    HumanMessagePromptTemplate.from_template(user_template),
]

brute_force_qa_prompt = ChatPromptTemplate.from_messages(brute_force_messages_rag)

####################
# prompt dictionary
prompt_dict = {"default": default_qa_prompt,
               "default_strawberry": default_strawberry_qa_prompt,
               "brute_force": brute_force_qa_prompt
               }
