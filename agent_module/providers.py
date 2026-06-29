"""
LLM provider factory for blob watcher & report generator.
Creates LangChain chat model instances from ModelConfig entries.
"""
import logging
import re
from typing import Any, List, Optional
from urllib.parse import urlparse
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger("providers")

class AnthropicAzureChatModel(BaseChatModel):
    """
    Thin LangChain BaseChatModel wrapper around AnthropicFoundry.
    Translates LangChain message lists to Anthropic Messages API calls.
    """
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 4096
    tools: Optional[List[dict]] = None
    _client: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AnthropicFoundry
            except ImportError as exc:
                raise ImportError(
                    "Package 'anthropic' is required for anthropic_azure provider. "
                    "Install it with: pip install anthropic"
                ) from exc
            object.__setattr__(self, "_client", AnthropicFoundry(
                api_key=self.api_key,
                base_url=self.base_url,
            ))
        return self._client

    def _convert_messages(self, messages: List[BaseMessage]):
        system_text: Optional[str] = None
        raw_anthropic_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_text = msg.content
                continue
                
            role = "user"
            content = []
            
            if isinstance(msg, HumanMessage):
                role = "user"
                content = [{"type": "text", "text": msg.content}]
            elif isinstance(msg, AIMessage):
                role = "assistant"
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
            else:
                role = "user"
                content = [{"type": "text", "text": str(msg.content)}]
                
            raw_anthropic_messages.append({"role": role, "content": content})
            
        merged_messages = []
        for msg in raw_anthropic_messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                last_content = merged_messages[-1]["content"]
                new_content = msg["content"]
                merged_messages[-1]["content"] = last_content + new_content
            else:
                merged_messages.append(msg)
                
        for msg in merged_messages:
            if len(msg["content"]) == 1 and msg["content"][0]["type"] == "text":
                msg["content"] = msg["content"][0]["text"]
                
        return system_text, merged_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._get_client()
        system_text, anthropic_messages = self._convert_messages(messages)

        create_kwargs: dict = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if stop:
            create_kwargs["stop_sequences"] = stop

        response = client.messages.create(**create_kwargs)
        
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
        
        text_content = self._clean_response(text_content)
        message = AIMessage(content=text_content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @property
    def _llm_type(self) -> str:
        return "anthropic_azure"


def create_llm(model_config):
    """
    Factory: create a LangChain chat model from a ModelConfig.
    """
    provider = model_config.provider
    
    if provider == "anthropic_azure":
        logger.info("Creating LLM [anthropic_azure] │ %s │ %s", model_config.name, model_config.deployment)
        base_url = model_config.endpoint.rstrip("/") + "/"
        if not base_url.endswith("anthropic/"):
            base_url = base_url.rstrip("/") + "/anthropic/"
        return AnthropicAzureChatModel(
            model=model_config.deployment,
            api_key=model_config.api_key,
            base_url=base_url,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        logger.info("Creating native ChatAnthropic │ %s │ %s", model_config.name, model_config.deployment)
        return ChatAnthropic(
            model=model_config.deployment,
            anthropic_api_key=model_config.api_key,
        )
    elif provider == "azure_openai":
        logger.info("Creating AzureChatOpenAI │ %s │ %s", model_config.name, model_config.deployment)
        return AzureChatOpenAI(
            azure_endpoint=model_config.endpoint,
            api_key=model_config.api_key,
            azure_deployment=model_config.deployment,
            api_version=model_config.api_version,
            temperature=0
        )
    elif provider == "openai":
        logger.info("Creating ChatOpenAI │ %s │ %s", model_config.name, model_config.deployment)
        return ChatOpenAI(
            api_key=model_config.api_key,
            model=model_config.deployment,
            temperature=0
        )
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please install langchain-google-genai to use Gemini models.")
        logger.info("Creating ChatGoogleGenerativeAI │ %s │ %s", model_config.name, model_config.deployment)
        return ChatGoogleGenerativeAI(
            model=model_config.deployment,
            google_api_key=model_config.api_key,
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported provider type '{provider}' for model '{model_config.name}' in report generator.")
