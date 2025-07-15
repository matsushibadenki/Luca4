# /app/constants.py
# title: アプリケーション定数
# role: プロジェクト全体で使用される定数を管理する。

from enum import Enum

class Thresholds:
    """評価スコアの閾値"""
    RELEVANCE_SCORE = 8
    COMPLETENESS_SCORE = 8

class MemoryInsightType(Enum):
    """メモリ洞察のタイプ"""
    PHYSICAL_SIMULATION = "physical_simulation_insight"
    SELF_CORRECTION = "self_correction_insight"

class ToolNames:
    """ツールの名前"""
    DYNAMIC_WEB_BROWSER = "DynamicWebBrowser"
    SEARCH = "WebSearch"
