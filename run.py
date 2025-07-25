# /run.py
# title: アプリケーション実行スクリプト
# role: メインAPIとアナリティクスサーバーを個別のポートで起動する。

import logging
import sys
import os
import uvicorn
import threading
from dotenv import load_dotenv
from typing import List, AsyncGenerator, cast
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import LlamaCpp


# プロジェクトのルートパスをシステムパスに追加
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

from app.containers import Container, wire_circular_dependencies
from app.config import settings
from app.utils.ollama_utils import check_ollama_models_availability
from app.utils.api_key_checker import check_search_api_key
from app.api import router as api_router
from app.analytics.router import router as analytics_router
from app.system_governor import SystemGovernor

# ロギングの基本設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# グローバルなDIコンテナのインスタンス化と循環依存の解決
container = Container()
wire_circular_dependencies(container)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    アプリケーションの起動と終了時に実行されるライフスパンイベント。
    """
    # --- 起動時 ---
    logger.info(f"--- Lifespan startup for {app.title} ---")
    app.state.container = container

    # LLMインスタンスの生成と注入はDIコンテナに任せるため、このブロックは不要
    
    # SystemGovernorはメインアプリでのみ起動
    if app.title == "Luca4 - Self-Evolving Metacognitive AI Framework":
        system_governor: SystemGovernor = container.system_governor()
        logger.info("システムガバナーを起動します...")
        system_governor.start()
        app.state.system_governor = system_governor
    
    yield

    # --- 終了時 ---
    logger.info(f"--- Lifespan shutdown for {app.title} ---")
    if hasattr(app.state, 'system_governor'):
        logger.info("システムガバナーを停止し、リソースを解放します...")
        app.state.system_governor.stop()
    
    logger.info(f"--- {app.title} server shutdown complete ---")

def create_main_app() -> FastAPI:
    """メインのAI応答APIサーバーを生成する。"""
    app = FastAPI(
        title="Luca4 - Self-Evolving Metacognitive AI Framework",
        description="An API for interacting with the Luca4 AI system.",
        version="1.0.0",
        lifespan=lifespan
    )
    app.include_router(api_router, prefix="/api")
    return app

def create_analytics_app() -> FastAPI:
    """アナリティクス配信用サーバーを生成する。"""
    app = FastAPI(
        title="Luca4 - Analytics Dashboard",
        description="Provides real-time analytics for the Luca4 system.",
        version="1.0.0",
        lifespan=lifespan
    )
    app.include_router(analytics_router, prefix="/api")
    app.mount("/dashboard", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """ルートURLからダッシュボードへリダイレクトする。"""
        return RedirectResponse("/dashboard/analytics.html")
        
    return app

def run_server(app: FastAPI, host: str, port: int):
    """指定された設定でuvicornサーバーを実行する関数。"""
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    # --- 起動前チェック ---
    if settings.LLM_BACKEND == "ollama":
        required_models: List[str] = [
            cast(str, settings.GENERATION_LLM_SETTINGS["model"]),
            settings.EMBEDDING_MODEL_NAME
        ]
        if not check_ollama_models_availability(required_models):
            logger.error("必要なOllamaモデルが利用できないため、アプリケーションを終了します。")
            sys.exit(1)
    elif settings.LLM_BACKEND == "llama_cpp":
        if not os.path.exists(settings.LAMA_CPP_MODEL_PATH):
            logger.error(f"LLM_BACKENDが'llama_cpp'に設定されていますが、指定されたモデルパスが見つかりません: {settings.LAMA_CPP_MODEL_PATH}")
            logger.error("アプリケーションを終了します。LAMA_CPP_MODEL_PATHが正しいGGUFモデルを指しているか確認してください。")
            sys.exit(1)
        else:
            logger.info(f"llama.cppモデル '{settings.LAMA_CPP_MODEL_PATH}' が検出されました。")
    else:
        logger.error(f"不明なLLM_BACKEND設定です: {settings.LLM_BACKEND}。アプリケーションを終了します。")
        sys.exit(1)

    check_search_api_key()
    
    if not os.path.exists("static"):
        os.makedirs("static")

    # --- アプリケーションの作成 ---
    main_app = create_main_app()
    analytics_app = create_analytics_app()

    # --- サーバーのスレッド起動 ---
    main_server_thread = threading.Thread(
        target=run_server,
        args=(main_app, "0.0.0.0", 8000),
        daemon=True
    )
    analytics_server_thread = threading.Thread(
        target=run_server,
        args=(analytics_app, "0.0.0.0", 8001),
        daemon=True
    )

    logger.info("--- Luca4 Main APIサーバーをポート8000で起動します ---")
    main_server_thread.start()
    logger.info("--- Luca4 Analyticsサーバーをポート8001で起動します ---")
    analytics_server_thread.start()

    try:
        main_server_thread.join()
        analytics_server_thread.join()
    except KeyboardInterrupt:
        logger.info("--- アプリケーション終了シグナル受信 ---")
    finally:
        logger.info("--- DIコンテナのリソースをシャットダウンします ---")
        container.shutdown_resources()
        logger.info("--- Luca4 全サーバー終了 ---")