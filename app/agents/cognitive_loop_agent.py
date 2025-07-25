# /app/agents/cognitive_loop_agent.py
# title: 認知ループAIエージェント
# role: 計画に基づき、情報検索、ツール利用、知識グラフ生成、そして概念操作を反復的に実行し、包括的な分析結果を生成する。

import logging
import re
import asyncio
from typing import Any, List, Dict, Set

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

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
from app.config import settings
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from app.reasoning.symbolic_verifier import SymbolicVerifier
from app.agents.deductive_reasoner_agent import DeductiveReasonerAgent
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

class CognitiveLoopAgent(AIAgent):
    """
    情報収集、評価、改善、概念操作を反復的に行い、知識を構造化する認知ループを実行するエージェント。
    """
    def __init__(
        self,
        llm: Any,
        output_parser: Any,
        prompt_template: ChatPromptTemplate,
        retriever: Retriever,
        retrieval_evaluator_agent: RetrievalEvaluatorAgent,
        query_refinement_agent: QueryRefinementAgent,
        knowledge_graph_agent: KnowledgeGraphAgent,
        persistent_knowledge_graph: PersistentKnowledgeGraph,
        tool_using_agent: ToolUsingAgent,
        tool_belt: ToolBelt,
        memory_consolidator: MemoryConsolidator,
        sensory_processing_unit: SensoryProcessingUnit,
        conceptual_memory: ConceptualMemory,
        imagination_engine: ImaginationEngine,
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        symbolic_verifier: SymbolicVerifier,
        deductive_reasoner_agent: DeductiveReasonerAgent,
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.retriever = retriever
        self.retrieval_evaluator_agent = retrieval_evaluator_agent
        self.query_refinement_agent = query_refinement_agent
        self.knowledge_graph_agent = knowledge_graph_agent
        self.persistent_knowledge_graph = persistent_knowledge_graph
        self.tool_using_agent = tool_using_agent
        self.tool_belt = tool_belt
        self.memory_consolidator = memory_consolidator
        self.sensory_processing_unit = sensory_processing_unit
        self.conceptual_memory = conceptual_memory
        self.imagination_engine = imagination_engine
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.symbolic_verifier = symbolic_verifier
        self.deductive_reasoner_agent = deductive_reasoner_agent
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.summarizer_prompt = ChatPromptTemplate.from_template(
            """以下のウェブページの内容を、ユーザーの質問に答える形で要約してください。

ユーザーの質問: {question}

ウェブページの内容:
{page_content}
---
要約:"""
        )
        self.summarizer_chain = self.summarizer_prompt | self.llm | self.output_parser
        super().__init__()

    def build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | self.output_parser

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def _symbolic_reasoning_loop(self, query: str, plan: str) -> str:
        """
        AlphaGeometryにインスパイアされた、LLMによる仮説生成と記号的検証を繰り返すループ。
        """
        logger.info("--- Symbolic Reasoning Loop START ---")
        known_facts: Set[str] = set() # 現在までに証明された事実
        reasoning_trace = f"初期問題: {query}\n計画: {plan}\n\n"

        max_iterations = 5 # 無限ループを防ぐ
        for i in range(max_iterations):
            logger.info(f"Symbolic Reasoning Iteration {i+1}/{max_iterations}")

            # 1. 仮説生成エージェント (LLM)
            # 現在の既知の事実から、次に追加すべき補助線や注目すべき点を提案させる
            hypothesis_prompt = ChatPromptTemplate.from_template(
                """あなたは独創的な数学者です。以下の問題とこれまでに分かっている事実に基づき、
                問題を解決するための次のステップとして、新しい仮説や補助的な構成（例：「点Aと点Cを結ぶ」）を一つだけ提案してください。

                問題: {query}
                既知の事実: {known_facts}
                ---
                新しい仮説/構成:
                """
            )
            hypothesis_chain = hypothesis_prompt | self.llm | self.output_parser
            new_hypothesis = await hypothesis_chain.ainvoke({"query": query, "known_facts": str(known_facts) or "なし"})
            
            reasoning_trace += f"ステップ {i+1}: [仮説] {new_hypothesis}\n"
            logger.info(f"Generated Hypothesis: {new_hypothesis}")
            known_facts.add(new_hypothesis)

            # 2. 記号的検証エンジン (Symbolic Verifier)
            # 新しい仮説を追加したことで、ルールから演繹できる新しい事実がないかチェック
            deduced_facts = self.symbolic_verifier.verify_and_deduce(known_facts)

            if deduced_facts:
                reasoning_trace += f"ステップ {i+1}: [演繹] {', '.join(deduced_facts)}\n"
                known_facts.update(deduced_facts)
            else:
                reasoning_trace += f"ステップ {i+1}: [演繹] 新たな事実は導出されませんでした。\n"

            # 3. 演繹的推論エージェント
            # 現在の全事実から、結論を導き出せるか試みる
            conclusion = self.deductive_reasoner_agent.invoke({
                "query": query,
                "known_facts": str(known_facts)
            })
            reasoning_trace += f"ステップ {i+1}: [推論] {conclusion}\n\n"
            
            # 結論に最終的な答えが含まれているかチェック（簡易的な終了条件）
            if "結論として" in conclusion or "証明された" in conclusion:
                logger.info("結論に達したため、記号的推論ループを終了します。")
                break
        else:
            logger.warning("最大反復回数に達しました。")
        
        logger.info("--- Symbolic Reasoning Loop END ---")
        return reasoning_trace
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    async def _iterative_retrieval(self, query: str) -> str:
        """
        検索、評価、クエリ改善を繰り返して情報の質を高める反復的検索を非同期で実行します。
        クエリにURLが含まれている場合、Playwrightツールを使用し、その内容を要約します。
        """
        url_pattern = re.compile(r'https?://\S+')
        match = url_pattern.search(query)

        if match:
            url = match.group(0)
            logger.info(f"クエリからURL '{url}' を検出しました。DynamicWebBrowserツールを使用します。")
            browser_tool = self.tool_belt.get_tool("DynamicWebBrowser")
            if browser_tool and hasattr(browser_tool, 'use_async'):
                question_part = url_pattern.sub("", query).strip()
                page_content = await browser_tool.use_async(url)
                
                max_length = 15000 
                if len(page_content) > max_length:
                    logger.warning(f"Webページの内容が長すぎるため、{max_length}文字に切り詰めます。")
                    page_content = page_content[:max_length]
                    
                logger.info("取得したWebページの内容を要約します...")
                summarizer_tool = self.tool_belt.get_tool("Specialist_Summarization_Expert")
                if summarizer_tool and hasattr(summarizer_tool, 'use_async'):
                    logger.info("要約専門家ツール 'Specialist_Summarization_Expert' を使用します。")
                    summary_query = f"ユーザーの質問: {question_part}\n\nウェブページの内容:\n{page_content}"
                    summary = await summarizer_tool.use_async(summary_query)
                else:
                    logger.info("要約専門家ツールが見つからないため、汎用要約チェーンを使用します。")
                    summary = await self.summarizer_chain.ainvoke({
                        "question": question_part,
                        "page_content": page_content
                    })
                return summary
            else:
                logger.warning("DynamicWebBrowserツールが見つからないか、非同期メソッドをサポートしていません。")
        
        max_iterations = settings.PIPELINE_SETTINGS["cognitive_loop"]["max_iterations"]
        current_query = query
        final_info = ""
        tool_used_this_cycle = False

        for i in range(max_iterations):
            logger.info(f"検索イテレーション {i+1}/{max_iterations}: クエリ='{current_query}'")
            
            docs: List[Document] = self.retriever.invoke(current_query)
            rag_retrieved_info = "\n\n".join([doc.page_content for doc in docs])

            eval_input = {"query": current_query, "retrieved_info": rag_retrieved_info}
            evaluation = self.retrieval_evaluator_agent.invoke(eval_input)
            
            logger.info(f"RAG検索品質の評価: {evaluation}")

            relevance = evaluation.get("relevance_score", 0)
            completeness = evaluation.get("completeness_score", 0)
            
            current_retrieved_info = rag_retrieved_info

            if relevance <= 8 or completeness <= 8:
                logger.info("RAG検索結果が不十分なため、外部ツールの利用を検討します。")
                
                available_tools_desc = self.tool_belt.get_tool_descriptions()
                tool_selection_input = {
                    "tools": available_tools_desc,
                    "task": f"「{current_query}」について、RAGで得られなかった情報を補完するために、最適なツールと検索クエリを選択してください。"
                }
                
                try:
                    tool_decision = self.tool_using_agent.invoke(tool_selection_input)
                    if ": " in tool_decision:
                        chosen_tool_name, tool_query_str = tool_decision.split(": ", 1)
                        chosen_tool_name = chosen_tool_name.strip()
                        tool_query_str = tool_query_str.strip()

                        chosen_tool = self.tool_belt.get_tool(chosen_tool_name)
                        if chosen_tool:
                            logger.info(f"ツール '{chosen_tool_name}' を使用して '{tool_query_str}' を検索します。")
                            if hasattr(chosen_tool, 'use_async'):
                                tool_result = await chosen_tool.use_async(tool_query_str)
                            else:
                                tool_result = chosen_tool.use(tool_query_str)
                            current_retrieved_info = f"{current_retrieved_info}\n\n--- 外部ツール ({chosen_tool_name}) からの情報 ---\n{tool_result}"
                            logger.info("外部ツールからの情報取得完了。")
                            tool_used_this_cycle = True
                        else:
                            logger.warning(f"選択されたツール '{chosen_tool_name}' が見つかりません。")
                    else:
                        logger.warning(f"ToolUsingAgentの出力形式が不正です: {tool_decision}")

                except Exception as e:
                    logger.error(f"ツール利用中にエラーが発生しました: {e}", exc_info=True)
            
            final_info = current_retrieved_info

            if (relevance > 8 and completeness > 8) or tool_used_this_cycle:
                logger.info("十分な品質の情報が得られたか、または外部ツールが利用されたため、検索を終了します。")
                break
            
            refine_input = {
                "query": query,
                "evaluation_summary": evaluation.get("summary", ""),
                "suggestions": evaluation.get("suggestions", "")
            }
            refined_query = self.query_refinement_agent.invoke(refine_input)
            logger.info(f"改善されたクエリ: '{refined_query}'")
            current_query = refined_query
        else:
            logger.warning("最大反復回数に達しました。現在の情報で処理を続行します。")

        return final_info

    async def _conceptual_operation(self, plan_step: str) -> str:
        """計画のステップに基づき、概念操作を実行する"""
        logger.info(f"概念操作を実行中: {plan_step}")
        synthesis_match = re.search(r"「(.+?)」と「(.+?)」の概念を合成", plan_step)
        
        if synthesis_match:
            concept_a_text = synthesis_match.group(1)
            concept_b_text = synthesis_match.group(2)
            
            logger.info(f"概念合成: '{concept_a_text}' + '{concept_b_text}'")
            
            vectors = self.sensory_processing_unit.encode_texts([concept_a_text, concept_b_text])
            if vectors.size == 0:
                return "概念のベクトル化に失敗しました。"
            
            new_vector = self.imagination_engine.combine_concepts(list(vectors), [1.0, 1.0])
            
            similar_concepts = self.conceptual_memory.search_similar_concepts(new_vector, k=3)
            
            analysis_result = (
                f"「{concept_a_text}」と「{concept_b_text}」の概念を合成した結果、"
                f"新しい抽象的な概念が生成されました。この新しい概念は、"
                f"「{'、'.join([c['metadata']['text'] for c in similar_concepts]) if similar_concepts else '未知の領域'})」"
                f"といった既存の概念と類似性を持っています。"
            )
            return analysis_result
            
        return "計画された概念操作を解釈または実行できませんでした。"

    async def ainvoke(self, input_data: Dict[str, Any] | str) -> str:
        """
        認知ループを非同期で実行し、最終的な分析結果を返します。
        計画に応じて概念操作や記号的推論も実行します。
        """
        if not isinstance(input_data, dict):
            raise TypeError("CognitiveLoopAgent expects a dictionary as input.")

        query = input_data.get("query", "")
        plan = input_data.get("plan", "")
        reasoning_instruction = input_data.get("reasoning_instruction", "")

        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # Planに特定のキーワードが含まれていれば、対応する特殊ループを起動
        if "記号的検証" in plan or "数学的証明" in plan:
            # 記号的推論ループを実行
            reasoning_trace = await self._symbolic_reasoning_loop(query, plan)
            final_retrieved_info = reasoning_trace
        elif "概念" in plan:
            # 概念操作ループを実行
            plan_steps = [step.strip() for step in plan.split('\n') if step.strip()]
            conceptual_results = []
            for step in plan_steps:
                if "概念" in step:
                    result = await self._conceptual_operation(step)
                    conceptual_results.append(f"【概念操作の結果】\n{result}")
            final_retrieved_info = "\n\n".join(conceptual_results)
            if not final_retrieved_info:
                final_retrieved_info = "概念操作を実行しましたが、有効な結果が得られませんでした。"
        else:
            # 通常の情報検索ループ
            final_retrieved_info = await self._iterative_retrieval(query)
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        knowledge_graph_summary = "知識グラフの生成に失敗しました。"
        try:
            if final_retrieved_info and not ("記号的検証" in plan or "数学的証明" in plan):
                logger.info("検索結果から知識グラフを生成しています...")
                kg_input = {"text_chunk": final_retrieved_info[:4000]}
                
                logger.debug(f"KnowledgeGraphAgentに渡される入力: {kg_input}")
                
                try:
                    new_knowledge_graph = await asyncio.wait_for(
                        asyncio.to_thread(self.knowledge_graph_agent.invoke, kg_input),
                        timeout=60.0
                    )
                    
                    if new_knowledge_graph and new_knowledge_graph.nodes:
                        self.persistent_knowledge_graph.merge(new_knowledge_graph)
                        self.persistent_knowledge_graph.save()
                        knowledge_graph_summary = self.persistent_knowledge_graph.get_summary()
                        logger.info("知識グラフの生成とマージが完了しました。")
                    else:
                        logger.warning("生成された知識グラフが空か無効です。")

                except asyncio.TimeoutError:
                    logger.warning("知識グラフの生成がタイムアウトしました(60秒)。処理をスキップします。")
                    knowledge_graph_summary = "知識グラフの生成がタイムアウトしたため、処理をスキップしました。"
            else:
                logger.info("検索結果が空か、記号的推論モードのため、知識グラフの生成をスキップします。")
                knowledge_graph_summary = "分析対象の情報がなかったか、記号的推論モードのため、知識グラフは生成されませんでした。"
        except Exception as e:
            logger.error(f"知識グラフの生成またはマージ中にエラーが発生しました: {e}", exc_info=True)
        
        physical_insights_logs = self.memory_consolidator.get_recent_insights("physical_simulation_insight", limit=3)
        physical_insights = "\n".join([log.get("synthesized_knowledge", "") for log in physical_insights_logs])
        if not physical_insights:
            physical_insights = "現在、物理シミュレーションから得られた特筆すべき洞察はありません。"
        
        final_input = {
            "query": query,
            "plan": plan,
            "long_term_memory_context": knowledge_graph_summary,
            "final_retrieved_info": final_retrieved_info,
            "physical_insights": physical_insights,
            "reasoning_instruction": reasoning_instruction,
        }

        if self._chain is None:
            raise RuntimeError("CognitiveLoopAgent's chain is not initialized.")
        return await self._chain.ainvoke(final_input)