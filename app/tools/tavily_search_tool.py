# /app/tools/tavily_search_tool.py
# title: Tavily検索ツール
# role: Tavilyを使用して、Web上の情報を検索する。

from app.tools.base import Tool
from langchain_tavily import TavilySearch
from app.constants import ToolNames

class TavilySearchTool(Tool):
    """
    Tavily Searchを実行するためのツール。
    """
    def __init__(self):
        self.name = ToolNames.SEARCH
        self.description = "最新の出来事、一般的な知識、特定のトピックについてインターネットで検索します。"
        self.api_wrapper = TavilySearch(max_results=5)

    def use(self, query: str) -> str:
        """
        指定されたクエリでWeb検索を実行し、結果を返す。
        """
        results = self.api_wrapper.invoke(query)
        # 結果がリスト形式で返ってくるため、文字列に変換する
        return str(results)