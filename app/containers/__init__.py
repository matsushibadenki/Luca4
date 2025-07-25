# /app/containers/__init__.py
# title: アプリケーションDIコンテナ
# role: 各AIエージェント、LLM、プロンプトテンプレート、およびその他の依存関係を定義し、提供する。

from __future__ import annotations
import os
import logging
from dependency_injector import containers, providers
from typing import Any, Iterator, cast
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import LlamaCpp

# --- Config and Utils ---
from app.config import settings
from app.llm_providers import LLMProvider, OllamaProvider, LlamaCppProvider

# --- Core Components ---
from app.prompts.manager import PromptManager
from app.analytics.collector import AnalyticsCollector
from app.rag.knowledge_base import KnowledgeBase
from app.knowledge_graph.persistent_knowledge_graph import PersistentKnowledgeGraph
from app.rag.retriever import Retriever
from app.memory.memory_consolidator import MemoryConsolidator
from app.memory.working_memory import WorkingMemory
from app.conceptual_reasoning import SensoryProcessingUnit, ConceptualMemory, ImaginationEngine

# --- Agents ---
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
from app.agents.performance_benchmark_agent import PerformanceBenchmarkAgent
from app.agents.autonomous_agent import AutonomousAgent
from app.agents.master_agent import MasterAgent
from app.agents.capability_mapper_agent import CapabilityMapperAgent
from app.agents.deductive_reasoner_agent import DeductiveReasonerAgent
from app.agents.thought_evaluator_agent import ThoughtEvaluatorAgent
from app.agents.tree_of_thoughts_agent import TreeOfThoughtsAgent
from app.agents.process_reward_agent import ProcessRewardAgent
from app.cognitive_modeling.predictive_coding_engine import PredictiveCodingEngine
from app.cognitive_modeling.world_model_agent import WorldModelAgent
from app.integrated_information_processing.integrated_information_agent import IntegratedInformationAgent
from app.internal_dialogue import DialogueParticipantAgent, MediatorAgent, ConsciousnessStagingArea
from app.reasoning.complexity_analyzer import ComplexityAnalyzer
from app.reasoning.symbolic_verifier import SymbolicVerifier
from app.meta_cognition import SelfCriticAgent, MetaCognitiveEngine
from app.problem_discovery.problem_discovery_agent import ProblemDiscoveryAgent

# --- MicroLLM and Tools ---
from app.micro_llm import MicroLLMCreator, MicroLLMManager
from app.tools.tool_belt import ToolBelt

# --- Simulation ---
from physical_simulation.simulation_manager import SimulationManager
from physical_simulation.results_analyzer import SimulationEvaluatorAgent
from physical_simulation.agents.ppo_agent import PPOAgent
from physical_simulation.environments.block_stacking_env import BlockStackingEnv

# --- Systems ---
from app.affective_system import AffectiveEngine, EmotionalResponseGenerator
from app.digital_homeostasis import IntegrityMonitor, EthicalMotivationEngine
from app.value_evolution import ValueEvaluator
from app.meta_intelligence.cognitive_energy.manager import CognitiveEnergyManager

# --- Pipelines ---
from app.pipelines import (
    FullPipeline, SimplePipeline, ParallelPipeline, QuantumInspiredPipeline,
    SpeculativePipeline, SelfDiscoverPipeline, InternalDialoguePipeline,
    MicroLLMExpertPipeline, ConceptualReasoningPipeline, TreeOfThoughtsPipeline
)

# --- Meta-Intelligence & Top-Level ---
from app.meta_intelligence import (
    SelfEvolvingSystem, EmergentIntelligenceNetwork, EvolvingValueSystem
)
from app.system_governor import SystemGovernor
from app.engine import MetaIntelligenceEngine, ResourceArbiter
from app.meta_intelligence.evolutionary_controller import EvolutionaryController


logger = logging.getLogger(__name__)

# --- Helper Functions for DI ---
def _knowledge_base_provider(source_file_path: str) -> Iterator[KnowledgeBase]:
    kb = KnowledgeBase.create_and_load(source_file_path=source_file_path)
    yield kb
    del kb

def _select_llm_provider(backend: str, llm_settings: dict, llama_cpp_path: str) -> LLMProvider:
    if backend == "ollama":
        logger.info("LLM_BACKEND: OllamaProviderを選択しました。")
        return OllamaProvider(host=settings.OLLAMA_HOST)
    elif backend == "llama_cpp":
        if not llama_cpp_path or not os.path.exists(llama_cpp_path):
            raise ValueError(f"Llama.cppモデルパスが無効または見つかりません: {llama_cpp_path}")
        logger.info(f"LLM_BACKEND: LlamaCppProviderを選択しました。モデル: {llama_cpp_path}")
        return LlamaCppProvider(
            model_path=llama_cpp_path,
            n_ctx=llm_settings["n_ctx"],
            n_batch=llm_settings["n_batch"],
            temperature=llm_settings["temperature"],
        )
    else:
        raise ValueError(f"不明なLLM_BACKEND設定 '{backend}' です。")

def _get_llm_instance(llm_settings: dict) -> Any:
    if settings.LLM_BACKEND == "ollama":
        return OllamaLLM(
            model=llm_settings["model"],
            temperature=llm_settings["temperature"],
            base_url=settings.OLLAMA_HOST,
        )
    elif settings.LLM_BACKEND == "llama_cpp":
        return LlamaCpp(
            model_path=settings.LAMA_CPP_MODEL_PATH,
            n_ctx=llm_settings["n_ctx"],
            n_batch=llm_settings["n_batch"],
            temperature=llm_settings["temperature"],
            n_gpu_layers=llm_settings.get("n_gpu_layers", 0),
            verbose=False,
        )
    raise ValueError(f"Unknown LLM_BACKEND: {settings.LLM_BACKEND}")


class Container(containers.DeclarativeContainer):
    """
    アプリケーション全体の依存関係を定義し、注入するための単一のDIコンテナ。
    """
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.main", "app.api", "run", "app.analytics.router"
        ]
    )

    # --- Core Providers ---
    analytics_collector: providers.Singleton[AnalyticsCollector] = providers.Singleton(AnalyticsCollector)
    prompt_manager: providers.Singleton[PromptManager] = providers.Singleton(PromptManager, file_path="data/prompts/prompts.json")
    llm_provider: providers.Singleton[LLMProvider] = providers.Singleton(
        _select_llm_provider,
        backend=settings.LLM_BACKEND,
        llm_settings=settings.GENERATION_LLM_SETTINGS,
        llama_cpp_path=settings.LAMA_CPP_MODEL_PATH,
    )
    llm_instance: providers.Singleton[OllamaLLM | LlamaCpp] = providers.Singleton(_get_llm_instance, llm_settings=settings.GENERATION_LLM_SETTINGS)
    verifier_llm_instance: providers.Singleton[OllamaLLM | LlamaCpp] = providers.Singleton(_get_llm_instance, llm_settings=settings.VERIFIER_LLM_SETTINGS)
    codestral_llm_instance: providers.Singleton[OllamaLLM | LlamaCpp] = providers.Singleton(_get_llm_instance, llm_settings=settings.CODESTRAL_LLM_SETTINGS)
    output_parser: providers.Singleton[StrOutputParser] = providers.Singleton(StrOutputParser)
    json_output_parser: providers.Singleton[JsonOutputParser] = providers.Singleton(JsonOutputParser)
    knowledge_base: providers.Resource[KnowledgeBase] = providers.Resource(_knowledge_base_provider, source_file_path=settings.KNOWLEDGE_BASE_SOURCE)
    persistent_knowledge_graph: providers.Singleton[PersistentKnowledgeGraph] = providers.Singleton(PersistentKnowledgeGraph, storage_path=settings.KNOWLEDGE_GRAPH_STORAGE_PATH)
    retriever: providers.Singleton[Retriever] = providers.Singleton(Retriever, knowledge_base=knowledge_base, persistent_knowledge_graph=persistent_knowledge_graph)
    memory_consolidator: providers.Singleton[MemoryConsolidator] = providers.Singleton(MemoryConsolidator, log_file_path=settings.MEMORY_LOG_FILE_PATH)
    working_memory: providers.Singleton[WorkingMemory] = providers.Singleton(WorkingMemory)
    sensory_processing_unit: providers.Singleton[SensoryProcessingUnit] = providers.Singleton(SensoryProcessingUnit, model_name='clip-ViT-B-32')
    conceptual_memory: providers.Singleton[ConceptualMemory] = providers.Singleton(ConceptualMemory, dimension=providers.Factory(lambda spu: spu.get_embedding_dimension(), spu=sensory_processing_unit))
    imagination_engine: providers.Factory[ImaginationEngine] = providers.Factory(ImaginationEngine)
    symbolic_verifier: providers.Singleton[SymbolicVerifier] = providers.Singleton(SymbolicVerifier)

    # --- MicroLLM and Tools ---
    micro_llm_creator: providers.Factory[MicroLLMCreator] = providers.Factory(MicroLLMCreator, llm_provider=llm_provider, knowledge_graph=persistent_knowledge_graph)
    micro_llm_manager: providers.Factory[MicroLLMManager] = providers.Factory(MicroLLMManager, llm_provider=llm_provider, creator=micro_llm_creator)
    tool_belt: providers.Singleton[ToolBelt] = providers.Singleton(ToolBelt, llm_provider=llm_provider, micro_llm_manager=micro_llm_manager)

    # --- System Providers ---
    energy_manager: providers.Singleton[CognitiveEnergyManager] = providers.Singleton(CognitiveEnergyManager)
    integrity_monitor: providers.Factory[IntegrityMonitor] = providers.Factory(IntegrityMonitor, llm=verifier_llm_instance, knowledge_graph=persistent_knowledge_graph, analytics_collector=analytics_collector)
    value_evaluator: providers.Singleton[ValueEvaluator] = providers.Singleton(ValueEvaluator, llm=verifier_llm_instance, output_parser=json_output_parser, analytics_collector=analytics_collector)
    affective_engine: providers.Singleton[AffectiveEngine] = providers.Singleton(AffectiveEngine, integrity_monitor=integrity_monitor, value_evaluator=value_evaluator)
    emotional_response_generator: providers.Factory[EmotionalResponseGenerator] = providers.Factory(EmotionalResponseGenerator, llm=llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("EMOTIONAL_RESPONSE_PROMPT"), pm=prompt_manager))
    ethical_motivation_engine: providers.Factory[EthicalMotivationEngine] = providers.Factory(EthicalMotivationEngine, integrity_monitor=integrity_monitor, value_evaluator=value_evaluator)

    # --- Agent Providers ---
    knowledge_graph_agent: providers.Factory[KnowledgeGraphAgent] = providers.Factory(KnowledgeGraphAgent, llm=llm_instance, prompt_template=providers.Factory(lambda pm: pm.get_prompt("KNOWLEDGE_GRAPH_AGENT_PROMPT"), pm=prompt_manager))
    tool_using_agent: providers.Factory[ToolUsingAgent] = providers.Factory(ToolUsingAgent, llm=llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("TOOL_USING_AGENT_PROMPT"), pm=prompt_manager))
    retrieval_evaluator_agent: providers.Factory[RetrievalEvaluatorAgent] = providers.Factory(RetrievalEvaluatorAgent, llm=llm_instance, prompt_template=providers.Factory(lambda pm: pm.get_prompt("RETRIEVAL_EVALUATOR_AGENT_PROMPT"), pm=prompt_manager))
    query_refinement_agent: providers.Factory[QueryRefinementAgent] = providers.Factory(QueryRefinementAgent, llm=llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("QUERY_REFINEMENT_AGENT_PROMPT"), pm=prompt_manager))
    planning_agent: providers.Factory[PlanningAgent] = providers.Factory(PlanningAgent, llm=llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("PLANNING_AGENT_PROMPT"), pm=prompt_manager))
    decompose_agent: providers.Factory[DecomposeAgent] = providers.Factory(DecomposeAgent, llm=llm_instance, output_parser=output_parser)
    critique_agent: providers.Factory[CritiqueAgent] = providers.Factory(CritiqueAgent, llm=verifier_llm_instance, output_parser=output_parser)
    synthesize_agent: providers.Factory[SynthesizeAgent] = providers.Factory(SynthesizeAgent, llm=llm_instance, output_parser=output_parser)
    integrated_information_agent: providers.Factory[IntegratedInformationAgent] = providers.Factory(IntegratedInformationAgent, llm=llm_instance, output_parser=output_parser)
    dialogue_participant_agent: providers.Factory[DialogueParticipantAgent] = providers.Factory(DialogueParticipantAgent, llm=llm_instance)
    mediator_agent: providers.Factory[MediatorAgent] = providers.Factory(MediatorAgent, llm=llm_instance)
    consciousness_staging_area: providers.Factory[ConsciousnessStagingArea] = providers.Factory(ConsciousnessStagingArea, llm=llm_instance, mediator_agent=mediator_agent)
    world_model_agent: providers.Factory[WorldModelAgent] = providers.Factory(WorldModelAgent, llm=llm_instance, knowledge_graph_agent=knowledge_graph_agent, persistent_knowledge_graph=persistent_knowledge_graph)
    predictive_coding_engine: providers.Factory[PredictiveCodingEngine] = providers.Factory(PredictiveCodingEngine, world_model_agent=world_model_agent, working_memory=working_memory, knowledge_graph_agent=knowledge_graph_agent, persistent_knowledge_graph=persistent_knowledge_graph)
    self_critic_agent: providers.Factory[SelfCriticAgent] = providers.Factory(SelfCriticAgent, llm=verifier_llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("SELF_CRITIC_AGENT_PROMPT"), pm=prompt_manager))
    meta_cognitive_engine: providers.Factory[MetaCognitiveEngine] = providers.Factory(MetaCognitiveEngine, self_critic_agent=self_critic_agent)
    problem_discovery_agent: providers.Factory[ProblemDiscoveryAgent] = providers.Factory(ProblemDiscoveryAgent, llm=llm_instance, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("PROBLEM_DISCOVERY_AGENT_PROMPT"), pm=prompt_manager))
    self_improvement_agent: providers.Factory[SelfImprovementAgent] = providers.Factory(SelfImprovementAgent, llm=llm_instance, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("SELF_IMPROVEMENT_AGENT_PROMPT"), pm=prompt_manager))
    self_correction_agent: providers.Factory[SelfCorrectionAgent] = providers.Factory(SelfCorrectionAgent, llm=llm_instance, memory_consolidator=memory_consolidator, micro_llm_manager=micro_llm_manager, prompt_manager=prompt_manager)
    autonomous_agent: providers.Factory[AutonomousAgent] = providers.Factory(AutonomousAgent, llm=llm_instance, output_parser=output_parser, memory_consolidator=memory_consolidator, knowledge_base=knowledge_base, tool_belt=tool_belt)
    consolidation_agent: providers.Factory[ConsolidationAgent] = providers.Factory(ConsolidationAgent, llm=llm_instance, output_parser=output_parser, knowledge_base=knowledge_base, knowledge_graph_agent=knowledge_graph_agent, memory_consolidator=memory_consolidator, persistent_knowledge_graph=persistent_knowledge_graph, prompt_manager=prompt_manager)
    knowledge_gap_analyzer: providers.Factory[KnowledgeGapAnalyzerAgent] = providers.Factory(KnowledgeGapAnalyzerAgent, llm=llm_instance, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("KNOWLEDGE_GAP_ANALYZER_PROMPT"), pm=prompt_manager), memory_consolidator=memory_consolidator, knowledge_graph=persistent_knowledge_graph)
    capability_mapper_agent: providers.Factory[CapabilityMapperAgent] = providers.Factory(CapabilityMapperAgent, llm=llm_instance, prompt_template=providers.Factory(lambda pm: pm.get_prompt("CAPABILITY_MAPPER_PROMPT"), pm=prompt_manager))
    complexity_analyzer: providers.Factory[ComplexityAnalyzer] = providers.Factory(ComplexityAnalyzer, llm=llm_instance)
    orchestration_agent: providers.Factory[OrchestrationAgent] = providers.Factory(OrchestrationAgent, llm_provider=llm_provider, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("ORCHESTRATION_PROMPT"), pm=prompt_manager), complexity_analyzer=complexity_analyzer, tool_belt=tool_belt)
    deductive_reasoner_agent: providers.Factory[DeductiveReasonerAgent] = providers.Factory(DeductiveReasonerAgent, llm=verifier_llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("DEDUCTIVE_REASONER_AGENT_PROMPT"), pm=prompt_manager))
    process_reward_agent: providers.Factory[ProcessRewardAgent] = providers.Factory(ProcessRewardAgent, llm=verifier_llm_instance, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("PROCESS_REWARD_PROMPT"), pm=prompt_manager))
    master_agent: providers.Factory[MasterAgent] = providers.Factory(
        MasterAgent,
        llm=llm_instance,
        output_parser=output_parser,
        prompt_template=providers.Factory(lambda pm: pm.get_prompt("MASTER_AGENT_PROMPT"), pm=prompt_manager),
        memory_consolidator=memory_consolidator,
        ethical_motivation_engine=ethical_motivation_engine,
        predictive_coding_engine=predictive_coding_engine,
        working_memory=working_memory,
        value_evaluator=value_evaluator,
        affective_engine=affective_engine,
        emotional_response_generator=emotional_response_generator,
        analytics_collector=analytics_collector,
        orchestration_agent=orchestration_agent, # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️  FIXED ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    )
    performance_benchmark_agent: providers.Factory[PerformanceBenchmarkAgent] = providers.Factory(PerformanceBenchmarkAgent, orchestration_agent=orchestration_agent)
    thought_evaluator_agent: providers.Factory[ThoughtEvaluatorAgent] = providers.Factory(ThoughtEvaluatorAgent, llm=verifier_llm_instance, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("THOUGHT_EVALUATOR_PROMPT"), pm=prompt_manager))
    tree_of_thoughts_agent: providers.Factory[TreeOfThoughtsAgent] = providers.Factory(TreeOfThoughtsAgent, llm=llm_instance, thought_evaluator=thought_evaluator_agent, prompt_template=providers.Factory(lambda pm: pm.get_prompt("THOUGHT_GENERATOR_PROMPT"), pm=prompt_manager))
    cognitive_loop_agent: providers.Factory[CognitiveLoopAgent] = providers.Factory(CognitiveLoopAgent, llm=llm_instance, output_parser=output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("COGNITIVE_LOOP_AGENT_PROMPT"), pm=prompt_manager), retriever=retriever, retrieval_evaluator_agent=retrieval_evaluator_agent, query_refinement_agent=query_refinement_agent, knowledge_graph_agent=knowledge_graph_agent, persistent_knowledge_graph=persistent_knowledge_graph, tool_using_agent=tool_using_agent, tool_belt=tool_belt, memory_consolidator=memory_consolidator, sensory_processing_unit=sensory_processing_unit, conceptual_memory=conceptual_memory, imagination_engine=imagination_engine, symbolic_verifier=symbolic_verifier, deductive_reasoner_agent=deductive_reasoner_agent)

    # --- Simulation Providers ---
    simulation_env: providers.Factory[BlockStackingEnv] = providers.Factory(BlockStackingEnv)
    ppo_agent: providers.Factory[PPOAgent] = providers.Factory(PPOAgent, state_dim=providers.Factory(lambda env: env.observation_space.shape[0], env=simulation_env), action_dim=providers.Factory(lambda env: env.action_space.shape[0], env=simulation_env), lr_actor=settings.RL_AGENT_SETTINGS["ppo"]["lr_actor"], lr_critic=settings.RL_AGENT_SETTINGS["ppo"]["lr_critic"], gamma=settings.RL_AGENT_SETTINGS["ppo"]["gamma"], K_epochs=settings.RL_AGENT_SETTINGS["ppo"]["K_epochs"], eps_clip=settings.RL_AGENT_SETTINGS["ppo"]["eps_clip"])
    simulation_evaluator_agent: providers.Factory[SimulationEvaluatorAgent] = providers.Factory(SimulationEvaluatorAgent, llm=llm_instance, output_parser=json_output_parser, prompt_template=providers.Factory(lambda pm: pm.get_prompt("SIMULATION_EVALUATOR_PROMPT"), pm=prompt_manager))
    simulation_manager: providers.Factory[SimulationManager] = providers.Factory(
        SimulationManager,
        evaluator_agent=simulation_evaluator_agent,
        rl_agent=ppo_agent,
        environment=simulation_env
    )
    
    # --- Pipeline Providers ---
    simple_pipeline: providers.Factory[SimplePipeline] = providers.Factory(SimplePipeline, llm=llm_instance, output_parser=output_parser, retriever=retriever, prompt_manager=prompt_manager)
    full_pipeline: providers.Factory[FullPipeline] = providers.Factory(FullPipeline, master_agent=master_agent, planning_agent=planning_agent, cognitive_loop_agent=cognitive_loop_agent, meta_cognitive_engine=meta_cognitive_engine, problem_discovery_agent=problem_discovery_agent, memory_consolidator=memory_consolidator, analytics_collector=analytics_collector)
    parallel_pipeline: providers.Factory[ParallelPipeline] = providers.Factory(ParallelPipeline, llm=llm_instance, output_parser=output_parser, cognitive_loop_agent_factory=cognitive_loop_agent.provider)
    quantum_inspired_pipeline: providers.Factory[QuantumInspiredPipeline] = providers.Factory(QuantumInspiredPipeline, llm=llm_instance, output_parser=output_parser, integrated_information_agent=integrated_information_agent)
    speculative_pipeline: providers.Factory[SpeculativePipeline] = providers.Factory(SpeculativePipeline, drafter_llm=llm_instance, verifier_llm=verifier_llm_instance, output_parser=output_parser)
    self_discover_pipeline: providers.Factory[SelfDiscoverPipeline] = providers.Factory(SelfDiscoverPipeline, planning_agent=planning_agent, decompose_agent=decompose_agent, critique_agent=critique_agent, synthesize_agent=synthesize_agent, cognitive_loop_agent=cognitive_loop_agent)
    internal_dialogue_pipeline: providers.Factory[InternalDialoguePipeline] = providers.Factory(InternalDialoguePipeline, dialogue_participant_agent=dialogue_participant_agent, consciousness_staging_area=consciousness_staging_area, integrated_information_agent=integrated_information_agent)
    micro_llm_expert_pipeline: providers.Factory[MicroLLMExpertPipeline] = providers.Factory(MicroLLMExpertPipeline, llm_provider=llm_provider, tool_using_agent=tool_using_agent, tool_belt=tool_belt)
    conceptual_reasoning_pipeline: providers.Factory[ConceptualReasoningPipeline] = providers.Factory(ConceptualReasoningPipeline, planning_agent=planning_agent, cognitive_loop_agent=cognitive_loop_agent, master_agent=master_agent)
    tree_of_thoughts_pipeline: providers.Factory[TreeOfThoughtsPipeline] = providers.Factory(TreeOfThoughtsPipeline, tree_of_thoughts_agent=tree_of_thoughts_agent)

    # --- Top-Level System Providers ---
    self_evolving_system: providers.Factory[SelfEvolvingSystem] = providers.Factory(
        SelfEvolvingSystem,
        meta_cognitive_engine=meta_cognitive_engine,
        self_improvement_agent=self_improvement_agent,
        self_correction_agent=self_correction_agent,
        analytics_collector=analytics_collector,
        process_reward_agent=process_reward_agent,
    )
    resource_arbiter: providers.Singleton[ResourceArbiter] = providers.Singleton(ResourceArbiter, energy_manager=energy_manager)
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
            conceptual_reasoning=conceptual_reasoning_pipeline,
            micro_llm_expert=micro_llm_expert_pipeline,
            tree_of_thoughts=tree_of_thoughts_pipeline,
        ),
        resource_arbiter=resource_arbiter
    )
    evolutionary_controller: providers.Factory[EvolutionaryController] = providers.Factory(EvolutionaryController, performance_benchmark_agent=performance_benchmark_agent, knowledge_gap_analyzer=knowledge_gap_analyzer, memory_consolidator=memory_consolidator, capability_mapper_agent=capability_mapper_agent, knowledge_graph=persistent_knowledge_graph)
    system_governor: providers.Singleton[SystemGovernor] = providers.Singleton(SystemGovernor, evolutionary_controller=evolutionary_controller, self_evolving_system=self_evolving_system, autonomous_agent=autonomous_agent, consolidation_agent=consolidation_agent, memory_consolidator=memory_consolidator, simulation_manager=simulation_manager, knowledge_gap_analyzer=knowledge_gap_analyzer, micro_llm_manager=micro_llm_manager, performance_benchmark_agent=performance_benchmark_agent, emergent_network=providers.Factory(EmergentIntelligenceNetwork, provider=llm_provider), value_system=providers.Factory(EvolvingValueSystem, provider=llm_provider), energy_manager=energy_manager)

def wire_circular_dependencies(container: Container) -> None:
    """
    循環参照を持つプロバイダの依存関係を解決する。
    """
    cast(providers.Factory[FullPipeline], container.full_pipeline).add_kwargs(
        self_evolving_system=container.self_evolving_system
    )
    cast(providers.Factory[PerformanceBenchmarkAgent], container.performance_benchmark_agent).add_kwargs(
        engine=container.engine
    )