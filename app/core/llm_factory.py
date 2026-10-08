"""LLM 工厂类"""

from langchain_openai import ChatOpenAI
from app.config import config


class LLMFactory:
    """DeepSeek OpenAI 兼容模型工厂"""

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        model = model or config.deepseek_model
        base_url = base_url or config.deepseek_api_base
        api_key = api_key or config.deepseek_api_key

        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

        extra_body = {}
        extra_body["stream"] = streaming

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body if extra_body else None,
        )

        return llm

# 全局 LLM 工厂实例
llm_factory = LLMFactory()
