# /app/idle_manager.py
# title: Idle Time Manager
# role: Manages the application's idle state and triggers background tasks.

import time
import logging
import threading
from typing import Optional, List, Dict, Any, Callable

from app.meta_intelligence.self_improvement.evolution import SelfEvolvingSystem
from app.agents.autonomous_agent import AutonomousAgent
from app.agents.consolidation_agent import ConsolidationAgent
from app.meta_intelligence.emergent.network import EmergentIntelligenceNetwork
from app.meta_intelligence.value_evolution.values import EvolvingValueSystem
from app.memory.memory_consolidator import MemoryConsolidator
from app.config import settings
from physical_simulation.simulation_manager import SimulationManager
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from app.agents.knowledge_gap_analyzer import KnowledgeGapAnalyzerAgent
from app.micro_llm.manager import MicroLLMManager
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

logger = logging.getLogger(__name__)

class IdleManager:
    """
    アプリケーションのアイドル状態を監視し、自己進化、自律思考、記憶整理などの
    バックグラウンドタスクを指定された間隔で起動する。
    """
    def __init__(
        self,
        self_evolving_system: SelfEvolvingSystem,
        autonomous_agent: AutonomousAgent,
        consolidation_agent: ConsolidationAgent,
        emergent_network: EmergentIntelligenceNetwork,
        value_system: EvolvingValueSystem,
        memory_consolidator: MemoryConsolidator,
        simulation_manager: SimulationManager,
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        knowledge_gap_analyzer: KnowledgeGapAnalyzerAgent,
        micro_llm_manager: MicroLLMManager,
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    ):
        """
        IdleManagerを初期化します。
        """
        self.self_evolving_system = self_evolving_system
        self.autonomous_agent = autonomous_agent
        self.consolidation_agent = consolidation_agent
        self.emergent_network = emergent_network
        self.value_system = value_system
        self.memory_consolidator = memory_consolidator
        self.simulation_manager = simulation_manager
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.knowledge_gap_analyzer = knowledge_gap_analyzer
        self.micro_llm_manager = micro_llm_manager
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        self._last_active_time: float = time.time()
        self._is_idle: bool = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self._last_run_times: Dict[str, float] = {
            "self_evolution": 0,
            "autonomous_cycle": 0,
            "consolidation_cycle": 0,
            "wisdom_synthesis": 0,
            "emergent_discovery": 0,
            "value_evolution": 0,
            "simulation_cycle": 0,
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
            "knowledge_gap_analysis": 0,
            # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        }

    def _monitor_loop(self):
        """
        アイドル状態を監視し、各バックグラウンドタスクをスケジュールに従って実行するループ。
        """
        logger.info("Idle monitor thread started.")
        while not self._stop_event.is_set():
            if self._is_idle:
                current_time = time.time()
                
                # --- 各タスクの実行判定 ---
                self._run_task_if_due("self_evolution", settings.IDLE_EVOLUTION_TRIGGER_SECONDS, self._run_self_evolution, current_time)
                self._run_task_if_due("autonomous_cycle", settings.AUTONOMOUS_CYCLE_INTERVAL_SECONDS, self._run_autonomous_cycle, current_time)
                self._run_task_if_due("consolidation_cycle", settings.CONSOLIDATION_CYCLE_INTERVAL_SECONDS, self._run_consolidation_cycle, current_time)
                self._run_task_if_due("wisdom_synthesis", settings.WISDOM_SYNTHESIS_INTERVAL_SECONDS, self._run_wisdom_synthesis, current_time)
                self._run_task_if_due("simulation_cycle", settings.SIMULATION_CYCLE_INTERVAL_SECONDS, self._run_simulation_cycle, current_time)
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                self._run_task_if_due("knowledge_gap_analysis", settings.MICRO_LLM_CREATION_INTERVAL_SECONDS, self._run_knowledge_gap_analysis, current_time)
                # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
                self._run_task_if_due("emergent_discovery", settings.WISDOM_SYNTHESIS_INTERVAL_SECONDS * 2, self._run_emergent_discovery, current_time)
                self._run_task_if_due("value_evolution", settings.WISDOM_SYNTHESIS_INTERVAL_SECONDS * 3, self._run_value_evolution, current_time)

            time.sleep(5)
        logger.info("Idle monitor thread stopped.")

    def _run_task_if_due(self, task_name: str, interval: int, task_function: Callable[[], None], current_time: float):
        """指定した間隔が経過していればタスクを実行するヘルパー関数。"""
        if current_time - self._last_run_times[task_name] > interval:
            logger.info(f"Idle time task '{task_name}' is due. Starting execution.")
            try:
                task_function()
            except Exception as e:
                logger.error(f"Error during idle task '{task_name}': {e}", exc_info=True)
            finally:
                self._last_run_times[task_name] = current_time
                logger.info(f"Idle time task '{task_name}' finished.")

    # --- 各タスクの実行メソッド ---
    def _run_self_evolution(self):
        import asyncio
        asyncio.run(self.self_evolving_system.analyze_own_performance())

    def _run_autonomous_cycle(self):
        self.autonomous_agent.run_autonomous_cycle()

    def _run_consolidation_cycle(self):
        self.consolidation_agent.run_consolidation_cycle()

    def _run_wisdom_synthesis(self):
        self.consolidation_agent.synthesize_deep_wisdom()
        
    def _run_emergent_discovery(self):
        import asyncio
        asyncio.run(self.emergent_network.discover_and_foster("complex and abstract problem solving"))

    def _run_value_evolution(self):
        import asyncio
        if hasattr(self.memory_consolidator, 'get_recent_events'):
            recent_experiences = self.memory_consolidator.get_recent_events(limit=10)
            if recent_experiences:
                asyncio.run(self.value_system.evolve_values(recent_experiences))
            else:
                logger.info("No recent experiences to evolve values from. Skipping.")
        else:
            logger.warning("method 'get_recent_events' not found in MemoryConsolidator. Skipping value evolution.")

    def _run_simulation_cycle(self):
        """物理シミュレーション学習サイクルを実行する。"""
        self.simulation_manager.run_simulation_cycle()

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _run_knowledge_gap_analysis(self):
        """
        プロアクティブに知識のギャップを分析し、必要であればマイクロLLMの作成をトリガーする。
        """
        identified_topic = self.knowledge_gap_analyzer.analyze_for_gaps()
        if identified_topic:
            logger.info(f"プロアクティブな知識ギャップ分析により、トピック '{identified_topic}' の強化が決定されました。")
            self.micro_llm_manager.run_creation_cycle(topic=identified_topic)
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

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
        logger.info("Stopping idle monitor thread...")
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
