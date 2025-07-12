# /app/main.py
# title: アプリケーションメインモジュール
# role: DIコンテナから依存関係を注入され、ユーザー入力と自律思考のメインループを管理する。

import logging
import queue
import threading
import time
from dependency_injector.wiring import inject, Provide

from app.containers import Container
from app.models import MasterAgentResponse
from app.idle_manager import IdleManager
from app.engine import MetaIntelligenceEngine
from app.agents.orchestration_agent import OrchestrationAgent

logger = logging.getLogger(__name__)

def user_input_thread(input_queue: queue.Queue[str]):
    """
    別スレッドでユーザーからの入力を待ち、キューに追加する。
    """
    while True:
        try:
            user_input = input()
            input_queue.put(user_input)
            if user_input.lower() == 'quit':
                break
        except (EOFError, KeyboardInterrupt):
            input_queue.put('quit')
            break

@inject
def main_loop(
    engine: MetaIntelligenceEngine = Provide[Container.engine],
    idle_manager: IdleManager = Provide[Container.idle_manager],
    orchestration_agent: OrchestrationAgent = Provide[Container.orchestration_agent],
):
    """
    ユーザー入力の処理とAIの自律思考を並行して実行するメインループ。
    """
    print("--- AI協調応答システム起動 ---")

    input_queue: queue.Queue[str] = queue.Queue()
    input_thread = threading.Thread(target=user_input_thread, args=(input_queue,), daemon=True)
    input_thread.start()

    print("\nシステム: 何かお手伝いできることはありますか？ (終了するには 'quit' と入力してください)")
    print("あなた: ", end="", flush=True)
    
    try:
        while True:
            try:
                # ユーザーからの入力を待つ間、アイドル状態に設定
                idle_manager.set_idle()
                user_input = input_queue.get(timeout=1.0) # 1秒ごとにタイムアウト
                
                # 入力があればビジー状態に設定
                idle_manager.set_busy()
                
                if user_input.lower() == 'quit':
                    print("\nシステム: ご利用ありがとうございました。")
                    break

                if user_input:
                    print("\nシステム: 考え中...")
                    
                    # OrchestrationAgentを使用して動的に実行モードを決定
                    logger.info("オーケストレーションエージェントによるモード選択を開始します...")
                    orchestration_decision = orchestration_agent.invoke({"query": user_input})
                    logger.info(f"選択されたモード: {orchestration_decision.get('chosen_mode')}, 理由: {orchestration_decision.get('reason')}")

                    # 決定されたモードでエンジンを実行
                    response: MasterAgentResponse = engine.run(query=user_input, orchestration_decision=orchestration_decision)
                    
                    print("\n--- 最終回答 ---")
                    print(response["final_answer"])
                    print("\n--- 自己評価 ---")
                    print(response["self_criticism"])
                    print("\n--- 潜在的な問題 ---")
                    print(response["potential_problems"])
                    
                    print("\nシステム: 何かお手伝いできることはありますか？ (終了するには 'quit' と入力してください)")
                    print("あなた: ", end="", flush=True)
            
            except queue.Empty:
                # ユーザー入力がない場合は、ループを継続してアイドル状態を維持
                continue

    except KeyboardInterrupt:
        print("\nシステム: 対話を中断します。")