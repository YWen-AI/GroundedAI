from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

rerank_template = """
Please rank the following context in order of relevance to the user question:
{context}

Please use the following format with 1. being the most relevant:
1. context number
2. context number
3. context number
...
"""

user_template = """
user question: {question}
"""

messages_rerank = [
    SystemMessagePromptTemplate.from_template(rerank_template),
    HumanMessagePromptTemplate.from_template(user_template),
]

default_qa_prompt = ChatPromptTemplate.from_messages(messages_rerank)