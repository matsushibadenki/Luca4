# /app/agents/consolidation_agent.py
# title: 記憶統合AIエージェント
# role: 脳の神経リプレイを模倣し、オフラインで短期記憶（ワーキングメモリ）の内容を分析・再構成し、長期記憶（知識グラフ）へと統合する。

import os
import json
import logging
from typing import Any, List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.agents.base import AIAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.memory.memory_consolidator import MemoryConsolidator
from app.rag.knowledge_base import KnowledgeBase
from app.knowledge_graph.models import KnowledgeGraph
from app.prompts.manager import PromptManager


logger = logging.getLogger(__name__)

class ConsolidationAgent(AIAgent):
    """
    ワーキングメモリの内容をナレッジベースに統合するオフラインエージェント。
    """
    def __init__(
        self,
        llm: Any,
        output_parser: Any,
        knowledge_base: KnowledgeBase,
        knowledge_graph_agent: KnowledgeGraphAgent,
        memory_consolidator: MemoryConsolidator,
        persistent_knowledge_graph: PersistentKnowledgeGraph,
        prompt_manager: PromptManager,
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_manager.get_prompt("CONSOLIDATION_AGENT_PROMPT")
        self.knowledge_base = knowledge_base
        self.knowledge_graph_agent = knowledge_graph_agent
        self.memory_consolidator = memory_consolidator
        self.persistent_knowledge_graph = persistent_knowledge_graph
        self.processed_sessions_log = "memory/processed_sessions.log"
        self.wisdom_synthesis_chain = prompt_manager.get_prompt("WISDOM_SYNTHESIS_PROMPT") | self.llm | self.output_parser
        super().__init__()

    def build_chain(self) -> Runnable:
        """
        知識を要約・構造化するためのチェーンを構築します。
        """
        return self.prompt_template | self.llm | self.output_parser

    def _get_unprocessed_sessions(self) -> List[str]:
        """
        未処理のワーキングメモリセッションファイルのリストを取得します。
        """
        session_dir = self.memory_consolidator.working_memory_log_dir
        if not os.path.exists(session_dir):
            return []
            
        processed_sessions = set()
        if os.path.exists(self.processed_sessions_log):
            with open(self.processed_sessions_log, "r", encoding="utf-8") as f:
                processed_sessions = set(line.strip() for line in f)

        all_sessions = set(f for f in os.listdir(session_dir) if f.endswith(".json"))
        unprocessed = list(all_sessions - processed_sessions)
        logger.info(f"{len(unprocessed)}件の未処理セッションが見つかりました。")
        return unprocessed

    def _mark_session_as_processed(self, session_file: str) -> None:
        """
        処理済みのセッションをログに記録します。
        """
        os.makedirs(os.path.dirname(self.processed_sessions_log), exist_ok=True)
        with open(self.processed_sessions_log, "a", encoding="utf-8") as f:
            f.write(session_file + "\n")

    def run_consolidation_cycle(self) -> None:
        """
        記憶の統合サイクルを1回実行します。
        """
        logger.info("--- 記憶統合サイクル開始 (オフライン) ---")
        unprocessed_files = self._get_unprocessed_sessions()

        if not unprocessed_files:
            logger.info("統合すべき新しいセッション記憶はありません。")
            logger.info("--- 記憶統合サイクル完了 ---")
            return

        session_file = unprocessed_files[0]
        session_path = os.path.join(self.memory_consolidator.working_memory_log_dir, session_file)
        
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"セッションファイル {session_path} の読み込みに失敗しました: {e}")
            self._mark_session_as_processed(session_file)
            return

        session_id = session_data.get('session_id', 'unknown_session')
        logger.info(f"セッション {session_id} の内容を統合中...")
        prediction_errors_str = json.dumps(session_data.get("prediction_errors", []), ensure_ascii=False, indent=2)
        
        if not prediction_errors_str or prediction_errors_str == '[]':
            logger.warning("セッションに統合すべき予測誤差がありません。")
            self._mark_session_as_processed(session_file)
            os.remove(session_path)
            return

        synthesis_input = {"prediction_errors": prediction_errors_str}
        synthesized_knowledge = self.invoke(synthesis_input)
        
        if not synthesized_knowledge or not synthesized_knowledge.strip():
            logger.warning("統合の結果、新しい知識は生成されませんでした。")
            self._mark_session_as_processed(session_file)
            os.remove(session_path)
            return

        logger.info("統合された知識から知識グラフを生成しています...")
        kg_input = {"text_chunk": synthesized_knowledge}
        new_knowledge_graph = self.knowledge_graph_agent.invoke(kg_input)
        
        if isinstance(new_knowledge_graph, KnowledgeGraph):
            self.persistent_knowledge_graph.merge(new_knowledge_graph)
            self.persistent_knowledge_graph.save()
            kg_string = new_knowledge_graph.to_string()
        else:
            kg_string = "知識グラフの生成に失敗しました。"
            logger.error(f"KnowledgeGraphAgent did not return a KnowledgeGraph object, but {type(new_knowledge_graph)}")


        new_documents = [
            Document(page_content=fact, metadata={"source": f"consolidated_from_{session_id}"}) 
            for fact in synthesized_knowledge.strip().split('\n') if fact.strip()
        ]
        if new_documents:
            self.knowledge_base.add_documents(new_documents)
            logger.info(f"{len(new_documents)}個の新しいドキュメントをFAISSナレッジベースに追加しました。")

        self.memory_consolidator.log_autonomous_thought(
            topic=f"consolidation_of_{session_id}",
            synthesized_knowledge=f"【統合された知識】\n{synthesized_knowledge}\n\n【生成された知識グラフ】\n{kg_string}"
        )

        self._mark_session_as_processed(session_file)
        os.remove(session_path)
        logger.info(f"セッション {session_file} の統合が完了し、ファイルが削除されました。")
        logger.info("--- 記憶統合サイクル完了 ---")

    def synthesize_deep_wisdom(self) -> None: # New method
        """
        長期知識グラフ全体からより深い知恵を合成し、ログに記録する。
        """
        logger.info("--- 知恵合成サイクル開始 (オフライン) ---")
        graph_summary = self.persistent_knowledge_graph.get_graph().to_string()
        
        if "知識グラフは空です" in graph_summary:
            logger.info("知識グラフが空のため、知恵合成をスキップします。")
            logger.info("--- 知恵合成サイクル完了 ---")
            return
            
        try:
            wisdom = self.wisdom_synthesis_chain.invoke({"knowledge_graph_summary": graph_summary})
            
            if wisdom and wisdom.strip():
                self.memory_consolidator.log_autonomous_thought(
                    topic="wisdom_synthesis",
                    synthesized_knowledge=f"【合成された知恵】\n{wisdom}"
                )
                logger.info("知恵の合成が完了し、ログに記録されました。")
            else:
                logger.warning("知恵の合成結果が空でした。")
        except Exception as e:
            logger.error(f"知恵合成中にエラーが発生しました: {e}", exc_info=True)
        
        logger.info("--- 知恵合成サイクル完了 ---")

    def invoke(self, input_data: Dict[str, Any] | str) -> str:
        if not isinstance(input_data, dict):
            raise TypeError("ConsolidationAgent expects a dictionary as input.")
        
        if self._chain is None:
            raise RuntimeError("ConsolidationAgent's chain is not initialized.")
        result: str = self._chain.invoke(input_data)
        return result