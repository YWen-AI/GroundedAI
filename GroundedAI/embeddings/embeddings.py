from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings

from dotenv import load_dotenv, find_dotenv

from GroundedAI.utils.config_utils import load_config

class EmbeddingFactory:
    def __init__(self, provider, callbacks=[]):
        
        load_dotenv(find_dotenv())
        config = load_config()["embedding_parameters"]

        self.provider = provider
        self.azure_deployment = config["AzureOpenAI"]["embedding_deployment_azure"] if self.provider == "AzureOpenAI" else None
        self.openai_model = config["OpenAI"]["embedding_model_openai"] if self.provider == "OpenAI" else None
        load_dotenv(find_dotenv())

    def create_model(self):
        if self.provider == "AzureOpenAI":
            if not self.azure_deployment:
                raise ValueError("azure_deployment must be provided for AzureOpenAI provider")
            return AzureOpenAIEmbeddings(
                azure_deployment=self.azure_deployment
            )
        elif self.provider == "OpenAI":
            if not self.openai_model:
                raise ValueError("openai_model must be provided for OpenAI provider")
            return OpenAIEmbeddings(
                model=self.openai_model,
            )
        else:
            raise ValueError("Invalid provider. Expected 'AzureOpenAI' or 'OpenAI'.")