# /app/tools/playwright_browser_tool.py
# title: Playwrightブラウザツール
# role: Playwrightを使用して指定されたURLのWebページをレンダリングし、そのコンテンツを抽出する。

import logging
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

from app.tools.base import Tool

logger = logging.getLogger(__name__)

class PlaywrightBrowserTool(Tool):
    """
    Playwrightを使用して動的にレンダリングされたWebページのコンテンツを取得するツール。
    """
    def __init__(self):
        self.name = "DynamicWebBrowser"
        self.description = "JavaScriptでレンダリングされたWebページを含む、指定されたURLのコンテンツを完全に取得します。静的なHTMLだけでなく、動的なサイトの情報を得るのに適しています。"

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    async def use_async(self, query: str) -> str:
        """
        指定されたURL（クエリ）のコンテンツを非同期で取得する。
        """
        url = query
        logger.info(f"PlaywrightBrowserTool: URL '{url}' のコンテンツを非同期で取得します。")
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, timeout=60000)
                content = await page.inner_text("body")
                await browser.close()
                logger.info(f"PlaywrightBrowserTool: URL '{url}' のコンテンツ取得に成功しました。")
                return content
        except PlaywrightTimeoutError:
            logger.error(f"PlaywrightBrowserTool: URL '{url}' の読み込みがタイムアウトしました。")
            return f"エラー: URL '{url}' の読み込みがタイムアウトしました（60秒）。ページが非常に重いか、存在しない可能性があります。"
        except Exception as e:
            logger.error(f"PlaywrightBrowserTool: URL '{url}' の処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            return f"エラー: URL '{url}' の処理中に予期せぬエラーが発生しました: {e}"

    def use(self, query: str) -> str:
        """
        同期的な呼び出しはサポートされていません。
        """
        raise NotImplementedError("PlaywrightBrowserToolは非同期でのみ使用可能です。use_asyncを使用してください。")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️