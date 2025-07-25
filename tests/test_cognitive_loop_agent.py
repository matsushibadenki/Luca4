# /tests/test_cognitive_loop_agent.py
# title: 認知ループエージェントユニットテスト
# role: アプリケーションのCognitiveLoopAgent層のユニットテスト

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, create_autospec
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

# テスト対象のモジュール
from app.agents.cognitive_loop_agent import CognitiveLoopAgent

# 依存関係の型ヒントのためにインポート
from app.agents.base import AIAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.query_refinement_agent import QueryRefinementAgent
from app.agents.retrieval_evaluator_agent import RetrievalEvaluatorAgent
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.rag.retriever import Retriever
from app.tools.tool_belt import ToolBelt
from app.agents.tool_using_agent import ToolUsingAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.conceptual_reasoning.sensory_processing_unit import SensoryProcessingUnit
from app.conceptual_reasoning.conceptual_memory import ConceptualMemory
from app.conceptual_reasoning.imagination_engine import ImaginationEngine
from app.reasoning.symbolic_verifier import SymbolicVerifier
from app.agents.deductive_reasoner_agent import DeductiveReasonerAgent

@pytest.fixture
def mock_dependencies():
    """CognitiveLoopAgentの初期化に必要な依存関係のモックを作成するフィクスチャ"""
    # Runnableと見なされるべきオブジェクトをモック
    mock_llm = create_autospec(Runnable, instance=True, spec_set=True)
    mock_output_parser = create_autospec(Runnable, instance=True, spec_set=True)
    mock_prompt_template = create_autospec(ChatPromptTemplate, instance=True, spec_set=True)
    
    # チェーンの構築（|演算子）をモック
    # `__or__`（|）が呼ばれたら、それ自体もRunnableなモックを返すように設定
    mock_chain = create_autospec(Runnable, instance=True, spec_set=True)
    mock_prompt_template.__or__.return_value = mock_chain
    mock_chain.__or__.return_value = mock_chain
    # 最終的なチェーンのainvokeもモックしておく
    mock_chain.ainvoke = AsyncMock(return_value="Final answer from main chain.")
    
    # 依存関係の辞書を作成
    dependencies = {
        "llm": mock_llm,
        "output_parser": mock_output_parser,
        "prompt_template": mock_prompt_template,
        "retriever": create_autospec(Retriever, instance=True, spec_set=True),
        "retrieval_evaluator_agent": create_autospec(RetrievalEvaluatorAgent, instance=True, spec_set=True),
        "query_refinement_agent": create_autospec(QueryRefinementAgent, instance=True, spec_set=True),
        "knowledge_graph_agent": create_autospec(KnowledgeGraphAgent, instance=True, spec_set=True),
        "persistent_knowledge_graph": create_autospec(PersistentKnowledgeGraph, instance=True, spec_set=True),
        "tool_using_agent": create_autospec(ToolUsingAgent, instance=True, spec_set=True),
        "tool_belt": create_autospec(ToolBelt, instance=True, spec_set=True),
        "memory_consolidator": create_autospec(MemoryConsolidator, instance=True, spec_set=True),
        "sensory_processing_unit": create_autospec(SensoryProcessingUnit, instance=True, spec_set=True),
        "conceptual_memory": create_autospec(ConceptualMemory, instance=True, spec_set=True),
        "imagination_engine": create_autospec(ImaginationEngine, instance=True, spec_set=True),
        "symbolic_verifier": create_autospec(SymbolicVerifier, instance=True, spec_set=True),
        "deductive_reasoner_agent": create_autospec(DeductiveReasonerAgent, instance=True, spec_set=True),
    }
    return dependencies

def test_initialization(mock_dependencies):
    """CognitiveLoopAgentが正しく初期化されることをテストする"""
    agent = CognitiveLoopAgent(**mock_dependencies)
    assert isinstance(agent, AIAgent)
    assert agent.llm == mock_dependencies["llm"]
    assert hasattr(agent, '_chain')
    assert hasattr(agent, 'summarizer_chain')

@pytest.mark.anyio
@patch('app.agents.cognitive_loop_agent.asyncio.to_thread', new_callable=AsyncMock)
async def test_ainvoke_normal_retrieval_flow(mock_to_thread, mock_dependencies):
    """通常の反復検索ループが正しく動作することをテストする"""
    agent = CognitiveLoopAgent(**mock_dependencies)
    
    # 依存モックの戻り値を設定
    mock_dependencies["retriever"].invoke.return_value = [Document(page_content="initial context")]
    mock_dependencies["retrieval_evaluator_agent"].invoke.return_value = {"relevance_score": 9, "completeness_score": 9}
    mock_dependencies["memory_consolidator"].get_recent_insights.return_value = []
    mock_to_thread.return_value = MagicMock() # knowledge_graph_agent.invokeのモック

    # 実行
    input_data = {"query": "What is AI?", "plan": "Search for information about AI"}
    result = await agent.ainvoke(input_data)

    # 検証
    assert result == "Final answer from main chain."
    mock_dependencies["retriever"].invoke.assert_called_once_with("What is AI?")
    agent._chain.ainvoke.assert_awaited_once()
    # 最終的な入力に検索結果が含まれていることを確認
    final_call_args = agent._chain.ainvoke.call_args[0][0]
    assert "initial context" in final_call_args["final_retrieved_info"]


@pytest.mark.anyio
@patch('app.agents.cognitive_loop_agent.asyncio.to_thread', new_callable=AsyncMock)
async def test_ainvoke_symbolic_reasoning_flow(mock_to_thread, mock_dependencies):
    """記号的推論ループが計画に基づいて起動されることをテストする"""
    agent = CognitiveLoopAgent(**mock_dependencies)
    
    # 記号的推論ループ自体をモック化
    with patch.object(agent, '_symbolic_reasoning_loop', new_callable=AsyncMock) as mock_symbolic_loop:
        mock_symbolic_loop.return_value = "Symbolic reasoning result"
        mock_dependencies["memory_consolidator"].get_recent_insights.return_value = []
        mock_to_thread.return_value = MagicMock()

        # 実行 (日本語のキーワードを含むplanを使用)
        input_data = {"query": "Prove that...", "plan": "これは数学的証明と記号的検証を必要とします。"}
        await agent.ainvoke(input_data)

        # 検証
        mock_symbolic_loop.assert_awaited_once_with("Prove that...", "これは数学的証明と記号的検証を必要とします。")
        # 最終的な入力に記号的推論の結果が含まれていることを確認
        final_call_args = agent._chain.ainvoke.call_args[0][0]
        assert "Symbolic reasoning result" in final_call_args["final_retrieved_info"]
        # 通常の検索ループが呼ばれていないことを確認
        mock_dependencies["retriever"].invoke.assert_not_called()


@pytest.mark.anyio
@patch('app.agents.cognitive_loop_agent.asyncio.to_thread', new_callable=AsyncMock)
async def test_ainvoke_conceptual_operation_flow(mock_to_thread, mock_dependencies):
    """概念操作が計画に基づいて起動されることをテストする"""
    agent = CognitiveLoopAgent(**mock_dependencies)
    
    # 概念操作メソッドをモック化
    with patch.object(agent, '_conceptual_operation', new_callable=AsyncMock) as mock_conceptual_op:
        mock_conceptual_op.return_value = "Conceptual operation result"
        mock_dependencies["memory_consolidator"].get_recent_insights.return_value = []
        mock_to_thread.return_value = MagicMock()
        
        # 実行 (日本語のキーワードを含むplanを使用)
        plan = "「A」と「B」の概念を合成する"
        input_data = {"query": "Synthesize concepts", "plan": plan}
        await agent.ainvoke(input_data)

        # 検証
        mock_conceptual_op.assert_awaited_once_with(plan)
        # 最終的な入力に概念操作の結果が含まれていることを確認
        final_call_args = agent._chain.ainvoke.call_args[0][0]
        assert "Conceptual operation result" in final_call_args["final_retrieved_info"]
        # 通常の検索ループが呼ばれていないことを確認
        mock_dependencies["retriever"].invoke.assert_not_called()