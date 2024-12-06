from GroundedAI.llms.llms import LLMFactory
from GroundedAI.prompt_templates.rag_prompt_templates import prompt_dict
from GroundedAI.chains.chains import ChainFactory
from langchain.schema.output_parser import StrOutputParser


def initialize_chain():
    model = LLMFactory(provider="OpenAI").create_model()
    prompt = prompt_dict["simple_chat_strawberry_llm"]

    output_parser = StrOutputParser()

    chain = ChainFactory(
        chain_type="simple_chat_llm_chain",
        prompt_template=prompt,
        llm=model,
        output_parser=output_parser
    ).create_chain()

    return chain


def start_prompting(chain, prompt):
    response = chain.invoke({"question": prompt})
    return response


def chat():
    chain = initialize_chain()
    # prompt = "We are working with an Aluminum-8.5wt% Magnesium alloy. The alloy exhibits corrosion in the nitric acid mass loss test. What could be the possible reason for this observation?"
    prompt = """
    given this CSV:
    Name, Age, Annual Income, Bank Account ID, Country, Birth Date
    John Doe, 37, 11223, ijdsifiojoif, USA, 1987-10-11
    Jane Smith, 23, 221, sdjwieuhf123, UK, 2001-01-01
    Carlos Gomez, 34, 9238,-, Mexico, 1990-10-31
    Li Wei, 125, 5674, dhsdjsndjwjd87, China, 1899-1-1
    Amira Hassan, 1014, 87766898,-, Egypt, 1010-10-11

    What is Amira's bank account ID?
    """

    response = start_prompting(chain, prompt)
    print(response)


if __name__ == "__main__":
    chat()
