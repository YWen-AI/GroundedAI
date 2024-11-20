import unittest
from unittest.mock import patch, MagicMock

from GroundedAI.llms.llms import LLMFactory


class TestLLMFactory(unittest.TestCase):
    @patch("GroundedAI.llms.llms.AzureChatOpenAI")  # Mock AzureChatOpenAI
    @patch("GroundedAI.llms.llms.ChatOpenAI")  # Mock ChatOpenAI
    def test_create_openai_model(self, mock_chat_openai, mock_azure_chat_openai):
        # Set up the mock for ChatOpenAI
        mock_chat_instance = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance

        # Create an OpenAI model instance with LLMFactory
        factory = LLMFactory(provider="OpenAI", temperature=0.7)
        model = factory.create_model()

        # Assert that ChatOpenAI was called with correct parameters
        mock_chat_openai.assert_called_once_with(model=factory.openai_model, temperature=1, callbacks=None)

        # Check that the returned model is the mock instance
        self.assertEqual(model, mock_chat_instance)

    @patch("GroundedAI.llms.llms.AzureChatOpenAI")  # Mock AzureChatOpenAI
    @patch("GroundedAI.llms.llms.ChatOpenAI")  # Mock ChatOpenAI
    def test_create_azure_model(self, mock_chat_openai, mock_azure_chat_openai):
        # Set up the mock for AzureChatOpenAI
        mock_azure_instance = MagicMock()
        mock_azure_chat_openai.return_value = mock_azure_instance

        # Create an AzureOpenAI model instance with LLMFactory
        factory = LLMFactory(provider="AzureOpenAI", temperature=0.5)
        model = factory.create_model()

        # Assert that AzureChatOpenAI was called with correct parameters
        mock_azure_chat_openai.assert_called_once_with(azure_deployment=factory.azure_deployment, temperature=0.5, callbacks=None)

        # Check that the returned model is the mock instance
        self.assertEqual(model, mock_azure_instance)

    def test_invalid_provider(self):
        # Test case for an invalid provider
        with self.assertRaises(ValueError) as context:
            factory = LLMFactory(provider="InvalidProvider")
            factory.create_model()
        self.assertEqual(str(context.exception), "Invalid provider. Expected 'AzureOpenAI' or 'OpenAI'.")


if __name__ == "__main__":
    unittest.main()
