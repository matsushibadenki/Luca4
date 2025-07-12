# /app/containers.py
# title: アプリケーションDIコンテナ
# role: 各AIエージェント、LLM、プロンプトテンプレート、およびその他の依存関係を定義し、提供する。

from __future__ import annotations
import logging # ここにloggingをインポート
import os # 修正: osモジュールをインポート
from dependency_injector import containers, providers
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import Dict, Any, TYPE_CHECKING, Optional, Iterator

from app.config import settings
import app.agents.prompts as prompts
from app.llm_providers import LLMProvider, OllamaProvider, LlamaCppProvider
from app.micro_llm import MicroLLMCreator, MicroLLMManager
from app.analytics import AnalyticsCollector
from app.agents.base import AIAgent
from app.agents.information_agent import InformationAgent
from app.agents.logical_agent import LogicalAgent
from app.agents.emotional_agent import EmotionalAgent
from app.agents.user_profiling_agent import UserProfilingAgent
from app.agents.master_agent import MasterAgent
from app.agents.fact_checking_agent import FactCheckingAgent
from app.agents.autonomous_agent import AutonomousAgent
from app.agents.word_learning_agent import WordLearningAgent
from app.agents.knowledge_assimilation_agent import KnowledgeAssimilationAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.cognitive_loop_agent import CognitiveLoopAgent
from app.agents.tool_using_agent import ToolUsingAgent
from app.agents.retrieval_evaluator_agent import RetrievalEvaluatorAgent
from app.agents.query_refinement_agent import QueryRefinementAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.consolidation_agent import ConsolidationAgent
from app.agents.thinking_modules import DecomposeAgent, CritiqueAgent, SynthesizeAgent
from app.agents.self_improvement_agent import SelfImprovementAgent
from app.agents.orchestration_agent import OrchestrationAgent
from app.agents.self_correction_agent import SelfCorrectionAgent
from app.agents.knowledge_gap_analyzer import KnowledgeGapAnalyzerAgent
from app.cognitive_modeling.predictive_coding_engine import PredictiveCodingEngine
from app.cognitive_modeling.world_model_agent import WorldModelAgent
from app.integrated_information_processing.integrated_information_agent import IntegratedInformationAgent
from app.internal_dialogue.dialogue_participant_agent import DialogueParticipantAgent
from app.internal_dialogue.mediator_agent import MediatorAgent
from app.internal_dialogue.consciousness_staging_area import ConsciousnessStagingArea
from app.digital_homeostasis import IntegrityMonitor, EthicalMotivationEngine
from app.reasoning.complexity_analyzer import ComplexityAnalyzer
from app.meta_cognition.self_critic_agent import SelfCriticAgent
from app.meta_cognition.meta_cognitive_engine import MetaCognitiveEngine
from app.memory.working_memory import WorkingMemory
from app.memory.memory_consolidator import MemoryConsolidator
from app.problem_discovery.problem_discovery_agent import ProblemDiscoveryAgent
from app.rag.knowledge_base import KnowledgeBase
from app.rag.retriever import Retriever
from app.tools.tool_belt import ToolBelt
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.value_evolution.value_evaluator import ValueEvaluator
from app.engine import MetaIntelligenceEngine
from app.pipelines.base import BasePipeline
from app.pipelines.simple_pipeline import SimplePipeline
from app.pipelines.full_pipeline import FullPipeline
from app.pipelines.parallel_pipeline import ParallelPipeline
from app.pipelines.quantum_inspired_pipeline import QuantumInspiredPipeline
from app.pipelines.speculative_pipeline import SpeculativePipeline
from app.pipelines.self_discover_pipeline import SelfDiscoverPipeline
from app.pipelines.internal_dialogue_pipeline import InternalDialoguePipeline
from app.pipelines.micro_llm_expert_pipeline import MicroLLMExpertPipeline
from app.meta_intelligence import (
    MetaIntelligence, MasterSystemConfig, CollectiveIntelligenceOrganizer,
    SelfEvolvingSystem, DynamicArchitecture, EmergentIntelligenceNetwork, EvolvingValueSystem
)
from app.idle_manager import IdleManager
from physical_simulation.simulation_manager import SimulationManager
from physical_simulation.results_analyzer import SimulationEvaluatorAgent
from physical_simulation.agents.ppo_agent import PPOAgent
from physical_simulation.environments.block_stacking_env import BlockStackingEnv
from app.affective_system.affective_engine import AffectiveEngine
from app.affective_system.emotional_response_generator import EmotionalResponseGenerator
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import LlamaCpp

if TYPE_CHECKING:
    from langchain_core.prompts import ChatPromptTemplate

# ロガーの初期化
logger = logging.getLogger(__name__)

def _knowledge_base_provider(source_file_path: str) -> Iterator[KnowledgeBase]:
    """DIコンテナのリソースとしてKnowledgeBaseを提供するためのジェネレータ関数"""
    kb = KnowledgeBase.create_and_load(source_file_path=source_file_path)
    yield kb
    del kb

def _select_llm_provider(backend: str, llm_settings: dict, llama_cpp_path: str) -> LLMProvider:
    """
    設定に基づいて適切なLLMProviderのインスタンスを選択して返すヘルパー関数。
    """
    if backend == "ollama":
        logger.info("LLM_BACKEND: OllamaProviderを選択しました。")
        return OllamaProvider(host=settings.OLLAMA_HOST)
    elif backend == "llama_cpp":
        if not llama_cpp_path or not os.path.exists(llama_cpp_path):
            logger.error(f"Llama.cppモデルパスが無効または見つかりません: {llama_cpp_path}。")
            raise ValueError(f"Llama.cppモデルパスが無効または見つかりません: {llama_cpp_path}")
        logger.info(f"LLM_BACKEND: LlamaCppProviderを選択しました。モデル: {llama_cpp_path}")
        return LlamaCppProvider(
            model_path=llama_cpp_path,
            n_ctx=llm_settings["n_ctx"],
            n_batch=llm_settings["n_batch"],
            temperature=llm_settings["temperature"],
        )
    else:
        logger.warning(f"不明なLLM_BACKEND設定 '{backend}' です。デフォルトでOllamaProviderを使用します。")
        return OllamaProvider(host=settings.OLLAMA_HOST)

class Container(containers.DeclarativeContainer):
    """
    アプリケーション全体の依存関係を定義し、注入するためのDIコンテナ。
    """
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.main", "app.api", "run", "app.analytics.router"
        ]
    )

    # --- 0. アナリティクス ---
    analytics_collector: providers.Singleton[AnalyticsCollector] = providers.Singleton(AnalyticsCollector)

    # --- 1. 基本コンポーネント ---
    llm_provider: providers.Singleton[LLMProvider] = providers.Singleton(
        _select_llm_provider,
        backend=settings.LLM_BACKEND,
        llm_settings=settings.GENERATION_LLM_SETTINGS,
        llama_cpp_path=settings.LAMA_CPP_MODEL_PATH,
    )

    # 外部で生成されたLLMインスタンスを受け取るためのプレースホルダー
    # LLMの具体的なインスタンスタイプは選択されたプロバイダーに依存
    llm_instance: providers.Object[Any] = providers.Object()
    verifier_llm_instance: providers.Object[Any] = providers.Object()

    output_parser: providers.Singleton[StrOutputParser] = providers.Singleton(StrOutputParser)
    json_output_parser: providers.Singleton[JsonOutputParser] = providers.Singleton(JsonOutputParser)
    
    knowledge_base: providers.Resource[KnowledgeBase] = providers.Resource(
        _knowledge_base_provider,
        source_file_path=settings.KNOWLEDGE_BASE_SOURCE,
    )
    
    persistent_knowledge_graph: providers.Singleton[PersistentKnowledgeGraph] = providers.Singleton(
        PersistentKnowledgeGraph, storage_path=settings.KNOWLEDGE_GRAPH_STORAGE_PATH
    )
    retriever: providers.Singleton[Retriever] = providers.Singleton(Retriever, knowledge_base=knowledge_base)
    memory_consolidator: providers.Singleton[MemoryConsolidator] = providers.Singleton(
        MemoryConsolidator, log_file_path=settings.MEMORY_LOG_FILE_PATH
    )
    working_memory: providers.Singleton[WorkingMemory] = providers.Singleton(WorkingMemory)

    # --- 2. マイクロLLM関連 ---
    micro_llm_creator: providers.Factory[MicroLLMCreator] = providers.Factory(
        MicroLLMCreator,
        llm_provider=llm_provider,
        knowledge_graph=persistent_knowledge_graph
    )
    micro_llm_manager: providers.Factory[MicroLLMManager] = providers.Factory(
        MicroLLMManager,
        llm_provider=llm_provider,
        creator=micro_llm_creator
    )
    tool_belt: providers.Singleton[ToolBelt] = providers.Singleton(
        ToolBelt,
        llm_provider=llm_provider,
        micro_llm_manager=micro_llm_manager
    )

    # --- 3. 物理シミュレーション関連 ---
    simulation_env: providers.Factory[BlockStackingEnv] = providers.Factory(BlockStackingEnv)
    ppo_agent: providers.Factory[PPOAgent] = providers.Factory(
        PPOAgent,
        state_dim=providers.Factory(lambda env: env.observation_space.shape[0], env=simulation_env),
        action_dim=providers.Factory(lambda env: env.action_space.shape[0], env=simulation_env),
        lr_actor=settings.RL_AGENT_SETTINGS["ppo"]["lr_actor"],
        lr_critic=settings.RL_AGENT_SETTINGS["ppo"]["lr_critic"],
        gamma=settings.RL_AGENT_SETTINGS["ppo"]["gamma"],
        K_epochs=settings.RL_AGENT_SETTINGS["ppo"]["K_epochs"],
        eps_clip=settings.RL_AGENT_SETTINGS["ppo"]["eps_clip"],
    )
    simulation_evaluator_agent: providers.Factory[SimulationEvaluatorAgent] = providers.Factory(
        SimulationEvaluatorAgent,
        llm=llm_instance,
        output_parser=json_output_parser,
        prompt_template=prompts.SIMULATION_EVALUATOR_PROMPT
    )
    simulation_manager: providers.Factory[SimulationManager] = providers.Factory(
        SimulationManager,
        evaluator_agent=simulation_evaluator_agent,
        knowledge_graph=persistent_knowledge_graph,
        memory_consolidator=memory_consolidator,
        rl_agent=ppo_agent,
        environment=simulation_env,
    )

    # --- 4. 感情システム ---
    affective_engine: providers.Factory[AffectiveEngine] = providers.Factory(
        AffectiveEngine,
        integrity_monitor=providers.Factory(
            IntegrityMonitor,
            llm=verifier_llm_instance,
            knowledge_graph=persistent_knowledge_graph,
            analytics_collector=analytics_collector
        ),
        value_evaluator=providers.Factory(
            ValueEvaluator,
            llm=verifier_llm_instance,
            output_parser=json_output_parser,
            analytics_collector=analytics_collector
        )
    )
    emotional_response_generator: providers.Factory[EmotionalResponseGenerator] = providers.Factory(
        EmotionalResponseGenerator,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=prompts.EMOTIONAL_RESPONSE_PROMPT
    )

    # --- 5. 各種エージェントの定義 ---
    knowledge_gap_analyzer: providers.Factory[KnowledgeGapAnalyzerAgent] = providers.Factory(
        KnowledgeGapAnalyzerAgent,
        llm=llm_instance,
        output_parser=json_output_parser,
        prompt_template=prompts.KNOWLEDGE_GAP_ANALYZER_PROMPT,
        memory_consolidator=memory_consolidator,
        knowledge_graph=persistent_knowledge_graph,
    )
    complexity_analyzer: providers.Factory[ComplexityAnalyzer] = providers.Factory(
        ComplexityAnalyzer,
        llm=llm_instance
    )
    tool_using_agent: providers.Factory[ToolUsingAgent] = providers.Factory(
        ToolUsingAgent,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=prompts.TOOL_USING_AGENT_PROMPT
    )
    knowledge_graph_agent: providers.Factory[KnowledgeGraphAgent] = providers.Factory(
        KnowledgeGraphAgent,
        llm=llm_instance,
        prompt_template=prompts.KNOWLEDGE_GRAPH_AGENT_PROMPT
    )
    retrieval_evaluator_agent: providers.Factory[RetrievalEvaluatorAgent] = providers.Factory(
        RetrievalEvaluatorAgent,
        llm=llm_instance,
        prompt_template=prompts.RETRIEVAL_EVALUATOR_AGENT_PROMPT
    )
    query_refinement_agent: providers.Factory[QueryRefinementAgent] = providers.Factory(
        QueryRefinementAgent,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=prompts.QUERY_REFINEMENT_AGENT_PROMPT
    )
    cognitive_loop_agent: providers.Factory[CognitiveLoopAgent] = providers.Factory(
        CognitiveLoopAgent,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=prompts.COGNITIVE_LOOP_AGENT_PROMPT,
        retriever=retriever,
        retrieval_evaluator_agent=retrieval_evaluator_agent,
        query_refinement_agent=query_refinement_agent,
        knowledge_graph_agent=knowledge_graph_agent,
        persistent_knowledge_graph=persistent_knowledge_graph,
        tool_using_agent=tool_using_agent,
        tool_belt=tool_belt,
        memory_consolidator=memory_consolidator,
    )
    self_critic_agent: providers.Factory[SelfCriticAgent] = providers.Factory(
        SelfCriticAgent, llm=verifier_llm_instance, output_parser=output_parser,
        prompt_template=prompts.SELF_CRITIC_AGENT_PROMPT
    )
    meta_cognitive_engine: providers.Factory[MetaCognitiveEngine] = providers.Factory(
        MetaCognitiveEngine, self_critic_agent=self_critic_agent
    )
    problem_discovery_agent: providers.Factory[ProblemDiscoveryAgent] = providers.Factory(
        ProblemDiscoveryAgent, llm=llm_instance, output_parser=json_output_parser,
        prompt_template=prompts.PROBLEM_DISCOVERY_AGENT_PROMPT
    )
    self_improvement_agent: providers.Factory[SelfImprovementAgent] = providers.Factory(
        SelfImprovementAgent,
        llm=llm_instance,
        output_parser=json_output_parser,
        prompt_template=prompts.SELF_IMPROVEMENT_AGENT_PROMPT
    )
    self_correction_agent: providers.Factory[SelfCorrectionAgent] = providers.Factory(
        SelfCorrectionAgent,
        llm=llm_instance,
        output_parser=output_parser,
        memory_consolidator=memory_consolidator,
        prompt_template=prompts.SELF_CORRECTION_AGENT_PROMPT,
        micro_llm_manager=micro_llm_manager,
    )
    self_evolving_system: providers.Factory[SelfEvolvingSystem] = providers.Factory(
        SelfEvolvingSystem,
        meta_cognitive_engine=meta_cognitive_engine,
        self_improvement_agent=self_improvement_agent,
        self_correction_agent=self_correction_agent,
        analytics_collector=analytics_collector,
    )
    autonomous_agent: providers.Factory[AutonomousAgent] = providers.Factory(
        AutonomousAgent,
        llm=llm_instance,
        output_parser=output_parser,
        memory_consolidator=memory_consolidator,
        knowledge_base=knowledge_base,
        tool_belt=tool_belt,
    )
    consolidation_agent: providers.Factory[ConsolidationAgent] = providers.Factory(
        ConsolidationAgent,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=prompts.CONSOLIDATION_AGENT_PROMPT,
        knowledge_base=knowledge_base,
        knowledge_graph_agent=knowledge_graph_agent,
        memory_consolidator=memory_consolidator,
        persistent_knowledge_graph=persistent_knowledge_graph,
    )
    world_model_agent: providers.Factory[WorldModelAgent] = providers.Factory(
        WorldModelAgent,
        llm=llm_instance,
        knowledge_graph_agent=knowledge_graph_agent,
        persistent_knowledge_graph=persistent_knowledge_graph,
    )
    predictive_coding_engine: providers.Factory[PredictiveCodingEngine] = providers.Factory(
        PredictiveCodingEngine,
        world_model_agent=world_model_agent,
        working_memory=working_memory,
        knowledge_graph_agent=knowledge_graph_agent,
        persistent_knowledge_graph=persistent_knowledge_graph,
    )
    integrated_information_agent: providers.Factory[IntegratedInformationAgent] = providers.Factory(
        IntegratedInformationAgent,
        llm=llm_instance,
        output_parser=output_parser
    )
    dialogue_participant_agent: providers.Factory[DialogueParticipantAgent] = providers.Factory(
        DialogueParticipantAgent,
        llm=llm_instance
    )
    mediator_agent: providers.Factory[MediatorAgent] = providers.Factory(
        MediatorAgent,
        llm=llm_instance
    )
    consciousness_staging_area: providers.Factory[ConsciousnessStagingArea] = providers.Factory(
        ConsciousnessStagingArea,
        llm=llm_instance,
        mediator_agent=mediator_agent
    )
    value_evaluator: providers.Factory[ValueEvaluator] = providers.Factory(
        ValueEvaluator,
        llm=verifier_llm_instance,
        output_parser=json_output_parser,
        analytics_collector=analytics_collector,
    )
    integrity_monitor: providers.Factory[IntegrityMonitor] = providers.Factory(
        IntegrityMonitor,
        llm=verifier_llm_instance,
        knowledge_graph=persistent_knowledge_graph,
        analytics_collector=analytics_collector,
    )
    ethical_motivation_engine: providers.Factory[EthicalMotivationEngine] = providers.Factory(
        EthicalMotivationEngine,
        integrity_monitor=integrity_monitor,
        value_evaluator=value_evaluator,
    )
    decompose_agent: providers.Factory[DecomposeAgent] = providers.Factory(DecomposeAgent, llm=llm_instance, output_parser=output_parser)
    critique_agent: providers.Factory[CritiqueAgent] = providers.Factory(CritiqueAgent, llm=verifier_llm_instance, output_parser=output_parser)
    synthesize_agent: providers.Factory[SynthesizeAgent] = providers.Factory(SynthesizeAgent, llm=llm_instance, output_parser=output_parser)
    planning_agent: providers.Factory[PlanningAgent] = providers.Factory(
        PlanningAgent, llm=llm_instance, output_parser=output_parser,
        prompt_template=prompts.PLANNING_AGENT_PROMPT
    )
    orchestration_agent: providers.Factory[OrchestrationAgent] = providers.Factory(
        OrchestrationAgent,
        llm_provider=llm_provider,
        output_parser=json_output_parser,
        prompt_template=prompts.ORCHESTRATION_PROMPT,
        complexity_analyzer=complexity_analyzer,
        tool_belt=tool_belt,
    )

    # --- 6. パイプラインとマスターエージェントの定義（循環参照を解消） ---

    # MasterAgentの定義：engineへの依存を削除
    master_agent: providers.Factory[MasterAgent] = providers.Factory(
        MasterAgent,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=prompts.MASTER_AGENT_PROMPT,
        memory_consolidator=memory_consolidator,
        ethical_motivation_engine=ethical_motivation_engine,
        predictive_coding_engine=predictive_coding_engine,
        working_memory=working_memory,
        value_evaluator=value_evaluator,
        orchestration_agent=orchestration_agent,
        affective_engine=affective_engine,
        emotional_response_generator=emotional_response_generator,
        analytics_collector=analytics_collector,
    )

    # FullPipelineの定義：MasterAgentに直接依存させる
    full_pipeline: providers.Factory[FullPipeline] = providers.Factory(
        FullPipeline,
        master_agent=master_agent, # 循環参照を解消
        planning_agent=planning_agent,
        cognitive_loop_agent=cognitive_loop_agent,
        meta_cognitive_engine=meta_cognitive_engine,
        problem_discovery_agent=problem_discovery_agent,
        memory_consolidator=memory_consolidator,
        self_evolving_system=self_evolving_system,
        analytics_collector=analytics_collector,
    )

    # 他のパイプラインの定義
    simple_pipeline: providers.Factory[SimplePipeline] = providers.Factory(
        SimplePipeline,
        llm=llm_instance,
        output_parser=output_parser,
        retriever=retriever  # 依存性を cognitive_loop_agent から retriever に変更
    )
    parallel_pipeline: providers.Factory[ParallelPipeline] = providers.Factory(
        ParallelPipeline,
        llm=llm_instance,
        output_parser=output_parser,
        cognitive_loop_agent_factory=cognitive_loop_agent.provider
    )
    quantum_inspired_pipeline: providers.Factory[QuantumInspiredPipeline] = providers.Factory(
        QuantumInspiredPipeline,
        llm=llm_instance,
        output_parser=output_parser,
        integrated_information_agent=integrated_information_agent,
    )
    speculative_pipeline: providers.Factory[SpeculativePipeline] = providers.Factory(
        SpeculativePipeline,
        drafter_llm=llm_instance,
        verifier_llm=verifier_llm_instance,
        output_parser=output_parser
    )
    self_discover_pipeline: providers.Factory[SelfDiscoverPipeline] = providers.Factory(
        SelfDiscoverPipeline,
        planning_agent=planning_agent,
        decompose_agent=decompose_agent,
        critique_agent=critique_agent,
        synthesize_agent=synthesize_agent,
        cognitive_loop_agent=cognitive_loop_agent
    )
    internal_dialogue_pipeline: providers.Factory[InternalDialoguePipeline] = providers.Factory(
        InternalDialoguePipeline,
        dialogue_participant_agent=dialogue_participant_agent,
        consciousness_staging_area=consciousness_staging_area,
        integrated_information_agent=integrated_information_agent
    )
    micro_llm_expert_pipeline: providers.Factory[MicroLLMExpertPipeline] = providers.Factory(
        MicroLLMExpertPipeline,
        llm_provider=llm_provider,
        tool_using_agent=tool_using_agent,
        tool_belt=tool_belt,
    )

    # --- 7. コアエンジン ---
    engine: providers.Singleton[MetaIntelligenceEngine] = providers.Singleton(
        MetaIntelligenceEngine,
        pipelines=providers.Dict(
            simple=simple_pipeline,
            full=full_pipeline,
            parallel=parallel_pipeline,
            quantum=quantum_inspired_pipeline,
            speculative=speculative_pipeline,
            self_discover=self_discover_pipeline,
            internal_dialogue=internal_dialogue_pipeline,
            micro_llm_expert=micro_llm_expert_pipeline,
        )
    )
    
    # --- 9. アイドルマネージャー ---
    idle_manager: providers.Singleton[IdleManager] = providers.Singleton(
        IdleManager,
        self_evolving_system=self_evolving_system,
        autonomous_agent=autonomous_agent,
        consolidation_agent=consolidation_agent,
        emergent_network=providers.Factory(EmergentIntelligenceNetwork, provider=llm_provider),
        value_system=providers.Factory(EvolvingValueSystem, provider=llm_provider),
        memory_consolidator=memory_consolidator,
        simulation_manager=simulation_manager,
        knowledge_gap_analyzer=knowledge_gap_analyzer,
        micro_llm_manager=micro_llm_manager,
    )
    
    # --- 10. MetaIntelligence (高度な機能) ---
    meta_intelligence_config: providers.Singleton[MasterSystemConfig] = providers.Singleton(MasterSystemConfig)
    meta_intelligence_system: providers.Singleton[MetaIntelligence] = providers.Singleton(
        MetaIntelligence,
        primary_provider=llm_provider,
        config=meta_intelligence_config
    )