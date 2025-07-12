# /app/pipelines/__init__.py
# title: 推論パイプラインパッケージ初期化ファイル
# role: このディレクトリをPythonのパッケージとして定義する。

from .base import BasePipeline
from .simple_pipeline import SimplePipeline
from .full_pipeline import FullPipeline
from .parallel_pipeline import ParallelPipeline
from .quantum_inspired_pipeline import QuantumInspiredPipeline
from .speculative_pipeline import SpeculativePipeline
from .self_discover_pipeline import SelfDiscoverPipeline
from .internal_dialogue_pipeline import InternalDialoguePipeline