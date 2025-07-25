# /app/system_governor.py
# title: System Governor
# role: Manages the application's state and triggers background tasks based on evolutionary goals.

import time
import logging
import threading
import asyncio
from typing import Optional, List, Dict, Any, Callable

from app.meta_intelligence.evolutionary_controller import EvolutionaryController
from app.meta_intelligence.self_improvement.evolution import SelfEvolvingSystem
from app.agents.autonomous_agent import AutonomousAgent
from app.agents.consolidation_agent import ConsolidationAgent
from app.meta_intelligence.emergent.network import EmergentIntelligenceNetwork
from app.meta_intelligence.value_evolution.values import EvolvingValueSystem
from app.memory.memory_consolidator import MemoryConsolidator
from app.config import settings
from physical_simulation.simulation_manager import SimulationManager
from app.agents.knowledge_gap_analyzer import KnowledgeGapAnalyzerAgent
from app.micro_llm.manager import MicroLLMManager
from app.agents.performance_benchmark_agent import PerformanceBenchmarkAgent
from app.meta_intelligence.cognitive_energy.manager import CognitiveEnergyManager

logger = logging.getLogger(__name__)

class SystemGovernor:
    """
    システムのアイドル状態を監視し、EvolutionaryControllerの方針に基づいて
    バックグラウンドタスクを動的に起動する。
    """
    def __init__(
        self,
        evolutionary_controller: EvolutionaryController,
        self_evolving_system: SelfEvolvingSystem,
        autonomous_agent: AutonomousAgent,
        consolidation_agent: ConsolidationAgent,
        emergent_network: EmergentIntelligenceNetwork,
        value_system: EvolvingValueSystem,
        memory_consolidator: MemoryConsolidator,
        simulation_manager: SimulationManager,
        knowledge_gap_analyzer: KnowledgeGapAnalyzerAgent,
        micro_llm_manager: MicroLLMManager,
        performance_benchmark_agent: PerformanceBenchmarkAgent,
        energy_manager: CognitiveEnergyManager,
    ):
        self.evolutionary_controller = evolutionary_controller
        self.self_evolving_system = self_evolving_system
        self.autonomous_agent = autonomous_agent
        self.consolidation_agent = consolidation_agent
        self.emergent_network = emergent_network
        self.value_system = value_system
        self.memory_consolidator = memory_consolidator
        self.simulation_manager = simulation_manager
        self.knowledge_gap_analyzer = knowledge_gap_analyzer
        self.micro_llm_manager = micro_llm_manager
        self.performance_benchmark_agent = performance_benchmark_agent
        self.energy_manager = energy_manager

        self._last_active_time: float = time.time()
        self._is_idle: bool = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self._last_run_times: Dict[str, float] = {
            "evolutionary_direction": 0,
            "performance_benchmark": 0,
            "self_evolution": 0,
            "knowledge_gap_analysis": 0,
            "consolidation_cycle": 0,
            "autonomous_cycle": 0,
            "wisdom_synthesis": 0,
            "simulation_cycle": 0,
            "emergent_discovery": 0,
            "value_evolution": 0,
        }
        self.current_goal: Optional[Dict[str, Any]] = None

    def _monitor_loop(self):
        """
        アイドル状態を監視し、各バックグラウンドタスクをスケジュールに従って実行するループ。
        """
        logger.info("System Governor monitor thread started.")
        while not self._stop_event.is_set():
            self.energy_manager._recover_energy()

            if self._is_idle:
                current_time = time.time()

                # 1. 進化の方向性を決定 (一定間隔で実行)
                if current_time - self._last_run_times["evolutionary_direction"] > settings.BENCHMARK_INTERVAL_SECONDS:
                    self.current_goal = asyncio.run(self.evolutionary_controller.determine_evolutionary_direction())
                    self._last_run_times["evolutionary_direction"] = current_time

                # 2. 現在の目標に基づいてタスクを実行
                if self.current_goal:
                    goal_type = self.current_goal.get("type")
                    if goal_type == "PERFORMANCE_IMPROVEMENT":
                        self._run_task_if_due("self_evolution", 60, self._run_self_evolution, current_time) # 60秒ごとに実行
                    elif goal_type == "KNOWLEDGE_ACQUISITION":
                        topic = self.current_goal.get("topic")
                        if topic:
                           self._run_task_if_due(f"micro_llm_{topic}", 3600, lambda: self._run_knowledge_gap_analysis(topic), current_time)
                    elif goal_type == "EXPLORATION":
                        self._run_task_if_due("autonomous_cycle", 120, self._run_autonomous_cycle, current_time) # 120秒ごとに実行
                
                # 3. 定期的なメンテナンス タスク
                self._run_task_if_due("consolidation_cycle", settings.CONSOLIDATION_CYCLE_INTERVAL_SECONDS, self._run_consolidation_cycle, current_time)
                self._run_task_if_due("wisdom_synthesis", settings.WISDOM_SYNTHESIS_INTERVAL_SECONDS, self._run_wisdom_synthesis, current_time)

            time.sleep(5)
        logger.info("System Governor monitor thread stopped.")

    def _run_task_if_due(self, task_name: str, interval: int, task_function: Callable[[], None], current_time: float):
        """指定した間隔が経過していればタスクを実行するヘルパー関数。"""
        if current_time - self._last_run_times.get(task_name, 0) > interval:
            logger.info(f"System Governor: Task '{task_name}' is due. Starting execution.")
            try:
                task_function()
            except Exception as e:
                logger.error(f"Error during Governor task '{task_name}': {e}", exc_info=True)
            finally:
                self._last_run_times[task_name] = current_time
                logger.info(f"System Governor: Task '{task_name}' finished.")

    # --- 各タスクの実行メソッド ---
    def _run_self_evolution(self):
        asyncio.run(self.self_evolving_system.analyze_own_performance())

    def _run_autonomous_cycle(self):
        self.autonomous_agent.run_autonomous_cycle()

    def _run_consolidation_cycle(self):
        self.consolidation_agent.run_consolidation_cycle()

    def _run_wisdom_synthesis(self):
        self.consolidation_agent.synthesize_deep_wisdom()

    def _run_knowledge_gap_analysis(self, topic: str):
        self.micro_llm_manager.run_creation_cycle(topic=topic)
        
    def _run_simulation_cycle(self):
        """物理シミュレーション学習サイクルを実行する。"""
        self.simulation_manager.run_simulation_cycle()

    def set_busy(self):
        if self._is_idle:
            logger.debug("System state changed to: Busy")
        self._is_idle = False
        self._last_active_time = time.time()

    def set_idle(self):
        if not self._is_idle:
            logger.debug("System state changed to: Idle")
        self._is_idle = True
        self._last_active_time = time.time()

    def start(self):
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

    def stop(self):
        logger.info("Stopping System Governor monitor thread...")
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()