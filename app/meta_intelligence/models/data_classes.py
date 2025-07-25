# /app/meta_intelligence/models/data_classes.py
# title: MetaIntelligenceデータクラス
# role: MetaIntelligenceシステム内で使用されるデータ構造を定義する。

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from app.meta_intelligence.consciousness.levels import ConsciousnessLevel

class ProblemClass(Enum):
    """
    問題の複雑度や性質を分類するための列挙型。
    """
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    TRANSCENDENT = "transcendent"

@dataclass
class ProblemSolution:
    """
    MetaIntelligenceによって解決された問題の解を表すデータクラス。
    """
    solution_content: str
    confidence: float
    problem_class: ProblemClass
    transcendence_achieved: bool
    processing_metadata: Dict[str, Any]
    emergent_insights: List[str] = field(default_factory=list)
    wisdom_generated: Optional[str] = None
    consciousness_evolution_triggered: bool = False
    integration_quality: float = 0.0

@dataclass
class MasterSystemConfig:
    """
    MetaIntelligenceマスターシステムの設定を保持するデータクラス。
    """
    enable_metacognition: bool = True
    enable_superintelligence: bool = True
    enable_dynamic_architecture: bool = True
    enable_value_evolution: bool = True
    enable_consciousness_evolution: bool = True
    auto_optimization: bool = True
    integration_depth: str = "full"  # "basic", "standard", "full", "transcendent"
    performance_monitoring: bool = True
    custom_parameters: Optional[Dict[str, Any]] = None

@dataclass
class IntegrationConfig:
    """
    Integration Orchestratorの設定を保持するデータクラス。
    """
    enable_all_systems: bool = True
    auto_evolution: bool = False
    integration_harmony_threshold: float = 0.8
    max_integration_depth: int = 10
    enable_emergent_insights: bool = True
    cross_system_communication: bool = True
    collective_memory_enabled: bool = True
    wisdom_synthesis_enabled: bool = True