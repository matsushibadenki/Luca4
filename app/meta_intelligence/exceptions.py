# /app/meta_intelligence/exceptions.py
# title: MetaIntelligenceカスタム例外
# role: MetaIntelligenceシステム内で使用されるカスタム例外を定義する。

class MetaIntelligenceError(Exception):
    """MetaIntelligenceシステムにおける一般的な基底例外クラス。"""
    pass

class InitializationError(MetaIntelligenceError):
    """システムの初期化中にエラーが発生した場合に送出される例外。"""
    pass

class ConfigurationError(MetaIntelligenceError):
    """設定に問題がある場合に送出される例外。"""
    pass

class ProblemSolvingError(MetaIntelligenceError):
    """問題解決プロセス中にエラーが発生した場合に送出される例外。"""
    pass