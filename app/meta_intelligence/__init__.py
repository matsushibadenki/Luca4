# /app/meta_intelligence/__init__.py
# title: MetaIntelligenceパッケージ
# role: 主要なAPIをトップレベルに公開する。

from .core.master_system import MetaIntelligence
from .core import IntegrationOrchestrator
from .models.data_classes import MasterSystemConfig, IntegrationConfig, ProblemSolution, ProblemClass
from .consciousness.levels import ConsciousnessLevel
from .meta_cognition.engine import MetaCognitionEngine
from .exceptions import MetaIntelligenceError, InitializationError
from .collective.organizer import CollectiveIntelligenceOrganizer
from .self_improvement.evolution import SelfEvolvingSystem
from .dynamic_architecture.architecture import DynamicArchitecture
from .emergent.network import EmergentIntelligenceNetwork
from .value_evolution.values import EvolvingValueSystem