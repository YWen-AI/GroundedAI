from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

from GroundedAI.utils.config_utils import load_config

class LLMFactory:
    def __init__(self, provider, temperature=0, callbacks=[]):

        load_dotenv(find_dotenv())
        config = load_config()["llm_parameters"]

        self.provider = provider
        self.temperature = temperature
        self.callbacks = callbacks
        self.azure_deployment = config["AzureOpenAI"]["llm_deployment_azure"] if self.provider == "AzureOpenAI" else None
        self.openai_model = config["OpenAI"]["llm_model_openai"] if self.provider == "OpenAI" else None

    def create_model(self):
        if self.provider == "AzureOpenAI":
            if not self.azure_deployment:
                raise ValueError("azure_deployment must be provided for AzureOpenAI provider")
            return AzureChatOpenAI(
                azure_deployment=self.azure_deployment,
                temperature=self.temperature,
                callbacks=self.callbacks
            )
        elif self.provider == "OpenAI":
            if not self.openai_model:
                raise ValueError("openai_model must be provided for OpenAI provider")
            if "o1" in self.openai_model:
                self.temperature = 1
            return ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                callbacks=self.callbacks
            )
        else:
            raise ValueError("Invalid provider. Expected 'AzureOpenAI' or 'OpenAI'.")
