# /app/meta_intelligence/evolutionary_controller.py
# title: 進化コントローラー
# role: システムの能力と知識の状態を監視し、次の進化の方向性を決定する最高意思決定機関。

import logging
from typing import Dict, Any, Optional
import json

from app.agents.performance_benchmark_agent import PerformanceBenchmarkAgent
from app.agents.knowledge_gap_analyzer import KnowledgeGapAnalyzerAgent
from app.memory.memory_consolidator import MemoryConsolidator
from app.agents.capability_mapper_agent import CapabilityMapperAgent
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph

logger = logging.getLogger(__name__)

class EvolutionaryController:
    """
    システムの進化戦略を決定するコントローラー。
    """
    def __init__(
        self,
        performance_benchmark_agent: PerformanceBenchmarkAgent,
        knowledge_gap_analyzer: KnowledgeGapAnalyzerAgent,
        memory_consolidator: MemoryConsolidator,
        capability_mapper_agent: CapabilityMapperAgent,
        knowledge_graph: PersistentKnowledgeGraph
    ):
        self.performance_benchmark_agent = performance_benchmark_agent
        self.knowledge_gap_analyzer = knowledge_gap_analyzer
        self.memory_consolidator = memory_consolidator
        self.capability_mapper_agent = capability_mapper_agent
        self.knowledge_graph = knowledge_graph
        self.current_evolutionary_goal: Optional[Dict[str, Any]] = None

    async def determine_evolutionary_direction(self) -> Optional[Dict[str, Any]]:
        """
        システムの現状を分析し、次の進化目標を決定する。
        """
        logger.info("--- Determining next evolutionary direction ---")

        # 1. パフォーマンスベンチマークを実行
        benchmark_report = await self.performance_benchmark_agent.run_benchmarks()
        overall_score = benchmark_report.get("summary", {}).get("overall_score", 0)

        # 2. 能力を知識グラフにマッピング
        try:
            logger.info("Mapping capabilities to knowledge graph...")
            capability_graph = self.capability_mapper_agent.invoke({
                "benchmark_report": json.dumps(benchmark_report, ensure_ascii=False, indent=2)
            })
            self.knowledge_graph.merge(capability_graph)
            self.knowledge_graph.save()
            logger.info("Successfully mapped capabilities to knowledge graph.")
        except Exception as e:
            logger.error(f"Failed to map capabilities to knowledge graph: {e}", exc_info=True)

        # 3. 知識ギャップを分析
        knowledge_gap = self.knowledge_gap_analyzer.analyze_for_gaps()

        # 4. 進化の方向性を決定
        # (このロジックは将来的にLLMによる高度な意思決定に置き換え可能)
        if overall_score < 0.7: # パフォーマンスに課題がある場合
            self.current_evolutionary_goal = {
                "type": "PERFORMANCE_IMPROVEMENT",
                "reason": f"Overall performance score ({overall_score:.2f}) is below the target threshold.",
                "details": "Focus on analyzing execution traces to improve pipeline efficiency.",
            }
        elif knowledge_gap: # 知識に偏りがある場合
            self.current_evolutionary_goal = {
                "type": "KNOWLEDGE_ACQUISITION",
                "reason": f"A knowledge gap was identified in the topic: '{knowledge_gap}'.",
                "details": f"Trigger Micro-LLM creation for the topic '{knowledge_gap}'.",
                "topic": knowledge_gap,
            }
        else:
            self.current_evolutionary_goal = {
                "type": "EXPLORATION",
                "reason": "System is stable. Focusing on autonomous research and wisdom synthesis.",
                "details": "Prioritize autonomous research and wisdom synthesis tasks.",
            }

        logger.info(f"New evolutionary goal set: {self.current_evolutionary_goal}")
        self.memory_consolidator.log_event("evolutionary_goal_set", self.current_evolutionary_goal)

        return self.current_evolutionary_goal