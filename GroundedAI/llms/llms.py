"""
This module provides a factory class, `LLMFactory`, for creating instances of language models from different
providers, now supporting Azure OpenAI and OpenAI. The `LLMFactory` class allows for configuration of model
parameters such as temperature and supports callback functions to customize responses.

Classes:
    LLMFactory: A class that initializes and creates LLMs based on the specified provider
    and settings.

Example:
    To create an instance of a language model using the `LLMFactory` class:

    ```python
    from llm_factory import LLMFactory

    # Initialize the factory with provider settings
    factory = LLMFactory(provider="OpenAI", temperature=0.7, callbacks=[my_callback_function])

    # Create the model instance ready to be prompt
    model = factory.create_model()
    ```
"""

from typing import List, Union, Optional, Callable

from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI

from GroundedAI.utils.config_utils import load_config


class LLMFactory:
    """
    A factory class to create instances of language models from different providers, such as Azure OpenAI or OpenAI,
    with configurable parameters.

    Args:
        provider (str): The LLM provider to use, either "AzureOpenAI" or "OpenAI".
        temperature (Union[int, float], optional): The temperature setting for the model, controlling response creativity.
            Defaults to 0.
        callbacks (Optional[List[Callable]], optional): List of callback functions to execute with model responses.
            Defaults to None.

    Attributes:
        provider (str): The LLM provider specified.
        temperature (Union[int, float]): Temperature value set for the model.
        callbacks (Optional[List[Callable]]): Callbacks associated with model responses.
        azure_deployment (Optional[str]): Azure deployment name for the AzureOpenAI provider, or None if not applicable.
        openai_model (Optional[str]): Model name for the OpenAI provider, or None if not applicable.

    Methods:
        create_model() -> Union[AzureChatOpenAI, ChatOpenAI]:
            Creates and returns an instance of the selected language model based on the provider.
            Raises a ValueError if required configurations for the provider are missing or if the provider is invalid.
    """

    def __init__(self, provider: str, temperature: Union[int, float] = 0, callbacks: Optional[List[Callable]] = None):
        config = load_config()["llm_parameters"]

        self.provider: str = provider
        self.temperature: Union[int, float] = temperature
        self.callbacks: Optional[List[Callable]] = callbacks
        self.azure_deployment: Optional[str] = config["AzureOpenAI"]["llm_deployment_azure"] if self.provider == "AzureOpenAI" else None
        self.openai_model: Optional[str] = config["OpenAI"]["llm_model_openai"] if self.provider == "OpenAI" else None

    def create_model(self) -> Union[AzureChatOpenAI, ChatOpenAI]:
        """
        Creates and returns an instance of a language model based on the specified provider.

        Returns:
            Union[AzureChatOpenAI, ChatOpenAI]: An instance of AzureChatOpenAI if the provider is "AzureOpenAI",
            or ChatOpenAI if the provider is "OpenAI".

        Raises:
            ValueError: If the required configurations for the specified provider are missing or the provider
            is invalid.
        """
        if self.provider == "AzureOpenAI":
            if not self.azure_deployment:
                raise ValueError("azure_deployment must be provided for AzureOpenAI provider")
            return AzureChatOpenAI(azure_deployment=self.azure_deployment, temperature=self.temperature, callbacks=self.callbacks)
        elif self.provider == "OpenAI":
            if not self.openai_model:
                raise ValueError("openai_model must be provided for OpenAI provider")
            if "o1" in self.openai_model:
                self.temperature = 1
            return ChatOpenAI(model=self.openai_model, temperature=self.temperature, callbacks=self.callbacks)
        else:
            raise ValueError("Invalid provider. Expected 'AzureOpenAI' or 'OpenAI'.")


if __name__ == "__main__":
    llm = LLMFactory(provider="OpenAI").create_model()
    print(llm)
