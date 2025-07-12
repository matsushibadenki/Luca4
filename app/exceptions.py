# /app/exceptions.py
# title: カスタム例外クラス
# role: アプリケーション固有のエラーを定義する。

class BaseAppException(Exception):
    """アプリケーションのすべてのカスタム例外の基底クラス。"""
    pass

class AgentError(BaseAppException):
    """エージェント関連のエラーの基底クラス。"""
    pass

class ChainInitializationError(AgentError):
    """エージェントのチェーン初期化に失敗した際のエラー。"""
    pass

class PipelineError(BaseAppException):
    """パイプライン実行中のエラーの基底クラス。"""
    pass

class PipelineExecutionError(PipelineError):
    """パイプラインの実行に失敗した際のエラー。"""
    pass

class ToolError(BaseAppException):
    """ツール関連のエラーの基底クラス。"""
    pass

class ToolNotFoundError(ToolError):
    """指定されたツールが見つからない場合のエラー。"""
    pass

class KnowledgeGraphError(BaseAppException):
    """知識グラフ関連のエラー。"""
    pass