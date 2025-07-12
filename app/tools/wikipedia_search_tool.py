# /app/tools/wikipedia_search_tool.py
# title: Wikipedia検索ツール
# role: Wikipediaから特定の記事を検索し、その要約を提供する。

import wikipedia
from app.tools.base import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class WikipediaSearchTool(Tool):
    """
    Wikipediaの記事を検索するためのツール。
    """
    def __init__(self):
        self.name = "WikipediaSearch"
        self.description = "特定の人物、場所、組織、概念に関する詳細な情報をWikipediaで検索します。"
        self.api_wrapper = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(wiki_client=wikipedia)
        )

    def use(self, query: str) -> str:
        """
        指定されたクエリでWikipediaを検索し、記事の要約を返す。
        """
        return self.api_wrapper.run(query)