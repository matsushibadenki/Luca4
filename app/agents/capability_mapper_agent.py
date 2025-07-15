# /app/agents/capability_mapper_agent.py
# title: 能力マッピングAIエージェント
# role: パフォーマンスベンチマークの結果を分析し、AIの能力を知識グラフ形式でマッピングする。

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from typing import Any, Dict

from app.agents.base import AIAgent
from app.knowledge_graph.models import KnowledgeGraph as KnowledgeGraphModel

class CapabilityMapperAgent(AIAgent):
    """
    ベンチマークレポートから能力の知識グラフを生成するAIエージェント。
    """
    def __init__(self, llm: Any, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_parser = JsonOutputParser(pydantic_object=KnowledgeGraphModel)
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        能力マッピングエージェントのLangChainチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def invoke(self, input_data: Dict[str, Any] | str) -> KnowledgeGraphModel:
        """
        ベンチマークレポートから知識グラフを生成して返します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("CapabilityMapperAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("CapabilityMapperAgent's chain is not initialized.")
        
        result_from_chain = self._chain.invoke(input_data)

        if isinstance(result_from_chain, dict):
            return KnowledgeGraphModel.model_validate(result_from_chain)
        elif isinstance(result_from_chain, KnowledgeGraphModel):
            return result_from_chain
        else:
            raise TypeError(f"CapabilityMapperAgentのチェーンから予期せぬ型が返されました: {type(result_from_chain)}")