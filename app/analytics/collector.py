# /app/analytics/collector.py
# title: アナリティクスデータ収集クラス
# role: システム全体の分析データを一元的に収集し、WebSocket経由での通知を管理する。

import asyncio
from typing import List, Dict, Any, Coroutine
from fastapi import WebSocket

class AnalyticsCollector:
    """
    シングルトンとして機能し、AIの各種分析データを収集・保持し、
    接続されているWebSocketクライアントにブロードキャストする。
    """
    _instance = None
    _active_connections: List[WebSocket] = []
    latest_data: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalyticsCollector, cls).__new__(cls)
        return cls._instance

    async def connect(self, websocket: WebSocket):
        """新しいWebSocket接続を登録する。"""
        await websocket.accept()
        self._active_connections.append(websocket)
        # 接続時に最新のデータを送信する
        await self.send_latest_data(websocket)

    def disconnect(self, websocket: WebSocket):
        """WebSocket接続を解除する。"""
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)

    async def send_latest_data(self, websocket: WebSocket):
        """指定されたWebSocketに最新の全データを送信する。"""
        if self.latest_data:
            await websocket.send_json(self.latest_data)

    async def _broadcast(self, data: Dict[str, Any]):
        """全ての接続済みクライアントにデータをブロードキャストする。"""
        # 最新データを更新
        self.latest_data.update(data)
        
        # 接続がない場合は何もしない
        if not self._active_connections:
            return

        # 非同期タスクのリストを作成
        tasks: List[Coroutine] = [
            connection.send_json(data) for connection in self._active_connections
        ]
        # タスクを並行して実行
        await asyncio.gather(*tasks, return_exceptions=True)

    async def log_event(self, event_type: str, data: Any):
        """
        任意のイベントをログに記録し、ブロードキャストする。
        
        Args:
            event_type (str): 'self_criticism', 'value_update' などのイベントタイプ。
            data (Any): 送信するデータ。
        """
        await self._broadcast({event_type: data})