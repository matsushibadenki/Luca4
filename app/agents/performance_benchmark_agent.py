# /app/agents/performance_benchmark_agent.py
# title: パフォーマンスベンチマークAIエージェント
# role: 標準化されたタスクを実行し、AIの性能（速度、精度、リソース）を測定・記録する。

import time
import logging
from typing import Any, Dict, List, Callable
import asyncio

from app.agents.base import AIAgent
from app.engine import MetaIntelligenceEngine
from app.agents.orchestration_agent import OrchestrationAgent
from app.models import MasterAgentResponse, OrchestrationDecision

logger = logging.getLogger(__name__)

# --- ベンチマークタスクの定義 ---

async def logical_puzzle_task(engine: MetaIntelligenceEngine, orchestration_agent: OrchestrationAgent) -> Dict[str, Any]:
    """論理パズルタスク"""
    query = "鶏、狼、そして穀物の袋を、狼と鶏、鶏と穀物を二人きりにしないようにして、川の向こう岸に渡す方法を教えてください。"
    expected_keywords = ["ボート", "一人ずつ", "鶏を連れて戻る"]
    
    # 複雑なタスクなので'full'モードを想定したOrchestrationDecisionをダミーで作成
    orchestration_decision: OrchestrationDecision = {
        "chosen_mode": "full",
        "reason": "Benchmark task: Logical Puzzle",
        "agent_configs": {},
        "reasoning_emphasis": "detail_oriented"
    }
    
    response = await engine.arun(query, orchestration_decision)
    
    # 精度チェック（簡易版）
    accuracy = sum(1 for keyword in expected_keywords if keyword in response["final_answer"]) / len(expected_keywords)
    return {"accuracy": accuracy, "final_answer": response["final_answer"]}

async def summarization_task(engine: MetaIntelligenceEngine, orchestration_agent: OrchestrationAgent) -> Dict[str, Any]:
    """長文要約タスク"""
    query = "https://arxiv.org/html/2506.16406v1 の内容を3つの箇条書きで要約してください。"
    expected_keywords = ["AlphaEvolve", "進化的探索", "コーディングエージェント"]

    # URLが含まれているため、OrchestrationAgentは'full'モードを選択すると期待される
    affective_state = None # ベンチマークでは感情状態は考慮しない
    orchestration_decision: OrchestrationDecision = orchestration_agent.invoke({"query": query, "affective_state": affective_state})

    response = await engine.arun(query, orchestration_decision)

    accuracy = sum(1 for keyword in expected_keywords if keyword in response["final_answer"].lower()) / len(expected_keywords)
    return {"accuracy": accuracy, "final_answer": response["final_answer"]}


# --- エージェントクラス ---

class PerformanceBenchmarkAgent(AIAgent):
    """
    システムのパフォーマンスを測定するための標準化されたベンチマークを実行するエージェント。
    """
    def __init__(self, engine: MetaIntelligenceEngine, orchestration_agent: OrchestrationAgent):
        self.engine = engine
        self.orchestration_agent = orchestration_agent
        self.benchmark_tasks: List[Dict[str, Any]] = [
            {"name": "Logical Puzzle", "task_func": logical_puzzle_task},
            {"name": "Summarization (URL)", "task_func": summarization_task},
        ]
        # このエージェントは直接的なチェーンを持たないため、初期化をオーバーライド
        self._chain = None

    def build_chain(self) -> None:
        return None

    async def run_benchmarks(self) -> Dict[str, Any]:
        """
        定義されたすべてのベンチマークタスクを実行し、結果を集計する。
        """
        logger.info("--- パフォーマンスベンチマークを開始します ---")
        overall_results = {}
        
        for task_info in self.benchmark_tasks:
            task_name = task_info["name"]
            task_func = task_info["task_func"]
            logger.info(f"ベンチマークタスク '{task_name}' を実行中...")
            
            start_time = time.time()
            
            try:
                task_result = await task_func(self.engine, self.orchestration_agent)
                execution_time = time.time() - start_time
                
                overall_results[task_name] = {
                    "execution_time_seconds": round(execution_time, 2),
                    "accuracy": round(task_result["accuracy"], 2),
                    "success": True,
                    "output_preview": task_result["final_answer"][:100] + "..."
                }
                logger.info(f"タスク '{task_name}' 完了。実行時間: {execution_time:.2f}s, 精度: {task_result['accuracy']:.2f}")

            except Exception as e:
                execution_time = time.time() - start_time
                overall_results[task_name] = {
                    "execution_time_seconds": round(execution_time, 2),
                    "accuracy": 0.0,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"タスク '{task_name}' の実行中にエラーが発生しました: {e}", exc_info=True)
        
        # 全体のサマリーを計算
        summary = self._summarize_results(overall_results)
        final_report = {"summary": summary, "details": overall_results}

        logger.info(f"--- パフォーマンスベンチマーク完了 ---")
        logger.info(f"総合スコア: {summary['overall_score']:.2f}, 平均実行時間: {summary['average_execution_time']:.2f}s")
        
        return final_report

    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """ベンチマーク結果を集計し、サマリーを生成する。"""
        total_tasks = len(results)
        successful_tasks = sum(1 for res in results.values() if res["success"])
        total_time = sum(res["execution_time_seconds"] for res in results.values())
        total_accuracy = sum(res["accuracy"] for res in results.values() if res["success"])

        avg_time = total_time / total_tasks if total_tasks > 0 else 0
        avg_accuracy = total_accuracy / successful_tasks if successful_tasks > 0 else 0
        
        # 総合スコア（精度と速度のバランスを考慮した簡易的なもの）
        # 速度ペナルティ: 平均時間が10秒を超えるとスコアが下がる
        time_penalty = max(0, (avg_time - 10) / 10) 
        overall_score = avg_accuracy * (1 - time_penalty)

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "average_execution_time": round(avg_time, 2),
            "average_accuracy": round(avg_accuracy, 2),
            "overall_score": round(overall_score, 2)
        }