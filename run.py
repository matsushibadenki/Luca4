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
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
from langchain_community.llms import LlamaCpp
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️


# プロジェクトのルートパスをシステムパスに追加
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

from app.containers import Container
from app.config import settings
from app.utils.ollama_utils import check_ollama_models_availability
from app.utils.api_key_checker import check_search_api_key
from app.api import router as api_router
from app.analytics.router import router as analytics_router
from app.idle_manager import IdleManager

# ロギングの基本設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# グローバルなDIコンテナ
container = Container()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    アプリケーションの起動と終了時に実行されるライフスパンイベント。
    """
    # --- 起動時 ---
    logger.info(f"--- Lifespan startup for {app.title} ---")
    app.state.container = container

    # LLMインスタンスを直接生成して注入
    logger.info("Creating and injecting LLM instances...")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    if settings.LLM_BACKEND == "ollama":
        llm_instance = OllamaLLM(
            model=settings.GENERATION_LLM_SETTINGS["model"],
            temperature=settings.GENERATION_LLM_SETTINGS["temperature"],
            base_url=settings.OLLAMA_HOST,
        )
        verifier_llm_instance = OllamaLLM(
            model=settings.VERIFIER_LLM_SETTINGS["model"],
            temperature=settings.VERIFIER_LLM_SETTINGS["temperature"],
            base_url=settings.OLLAMA_HOST,
        )
    elif settings.LLM_BACKEND == "llama_cpp":
        llm_instance = LlamaCpp(
            model_path=settings.LAMA_CPP_MODEL_PATH,
            n_ctx=settings.GENERATION_LLM_SETTINGS["n_ctx"],
            n_batch=settings.GENERATION_LLM_SETTINGS["n_batch"],
            temperature=settings.GENERATION_LLM_SETTINGS["temperature"],
            n_gpu_layers=settings.GENERATION_LLM_SETTINGS["n_gpu_layers"], # 追加
            verbose=False,
        )
        verifier_llm_instance = LlamaCpp(
            model_path=settings.LAMA_CPP_MODEL_PATH, # 検証LLMも同じモデルパスを共有すると仮定
            n_ctx=settings.VERIFIER_LLM_SETTINGS["n_ctx"],
            n_batch=settings.VERIFIER_LLM_SETTINGS["n_batch"],
            temperature=settings.VERIFIER_LLM_SETTINGS["temperature"],
            n_gpu_layers=settings.VERIFIER_LLM_SETTINGS["n_gpu_layers"], # 追加
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {settings.LLM_BACKEND}")
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    container.llm_instance.override(llm_instance)
    container.verifier_llm_instance.override(verifier_llm_instance)
    logger.info("LLM instances successfully injected.")

    # 手動での設定を削除し、DIコンテナのワイヤリングのみ行う
    container.wire(modules=[
        sys.modules[__name__], 
        "app.api", 
        "app.analytics.router"
    ])

    # アイドルマネージャーはメインアプリでのみ起動
    if app.title == "Luca4 - Self-Evolving Metacognitive AI Framework":
        idle_manager: IdleManager = container.idle_manager()
        logger.info("アイドルマネージャーを起動します...")
        idle_manager.start()
        app.state.idle_manager = idle_manager
    
    yield # ここでアプリケーションが実行される

    # --- 終了時 ---
    logger.info(f"--- Lifespan shutdown for {app.title} ---")
    if hasattr(app.state, 'idle_manager'):
        logger.info("アイドルマネージャーを停止し、リソースを解放します...")
        app.state.idle_manager.stop()
    
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
        # llama.cppの場合は、モデルファイルの存在確認など、別のチェックが必要になる可能性がある
        # ここでは、LAMA_CPP_MODEL_PATHが設定されていることを簡易的に確認
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
        # メインスレッドでスレッドの終了を待つ
        main_server_thread.join()
        analytics_server_thread.join()
    except KeyboardInterrupt:
        logger.info("--- アプリケーション終了シグナル受信 ---")
    finally:
        # アプリケーション終了時にリソースをクリーンアップ
        logger.info("--- DIコンテナのリソースをシャットダウンします ---")
        container.shutdown_resources()
        logger.info("--- Luca4 全サーバー終了 ---")