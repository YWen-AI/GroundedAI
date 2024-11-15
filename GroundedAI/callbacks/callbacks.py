from typing import List, Dict, Any
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from langchain_core.callbacks.base import BaseCallbackHandler

class LLMAppCallbackHandler(BaseCallbackHandler):
    """
    A callback handler for tracking the execution of chains and chat models.
    
    Attributes:
        session_id (str): The ID of the session.
        logger (logging.Logger): Logger for logging information with the session ID.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the LLMAppCallbackHandler with a session ID.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """ 
        super().__init__(*args, **kwargs)
        print("LLMAppCallbackHandler initialized.")


    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """
        Logs the start of a chain execution.

        Args:
            serialized (Dict[str, Any]): Serialized chain information.
            inputs (Dict[str, Any]): Inputs to the chain.
            **kwargs: Additional keyword arguments.
        """
        print(f"Chain {serialized['id'][-1]} started.")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """
        Logs the end of a chain execution.

        Args:
            outputs (Dict[str, Any]): Outputs of the chain.
            **kwargs: Additional keyword arguments.
        """
        print(f"Chain ended.")


    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs) -> None:
        """
        Logs the start of a chat model execution.

        Args:
            serialized (Dict[str, Any]): Serialized chat model information.
            messages (List[List[BaseMessage]]): Messages to the chat model.
            **kwargs: Additional keyword arguments.
        """
        print(f"Chat model {serialized['id'][-1]} started.")
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Logs the end of a chat model execution.

        Args:
            response (LLMResult): The response from the chat model.
            **kwargs: Additional keyword arguments.
        """        
        print(f"Chat model ended.")
