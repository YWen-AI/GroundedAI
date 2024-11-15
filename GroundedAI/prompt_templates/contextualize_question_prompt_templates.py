from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

user_template = """
{question}
"""

human_message_template = HumanMessagePromptTemplate.from_template(user_template)

contextualize_q_system_template = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Chat history:
{chat_history}

"""

contextualize_q_system_message_template = SystemMessagePromptTemplate.from_template(contextualize_q_system_template)

messages_contextualize_q = [
        contextualize_q_system_message_template,
        human_message_template
    ]

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    messages_contextualize_q
)