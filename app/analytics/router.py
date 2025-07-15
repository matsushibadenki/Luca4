# /app/analytics/router.py
# title: アナリティクスAPIルーター
# role: アナリティクスデータ配信用WebSocketエンドポイントを定義する。

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from dependency_injector.wiring import inject, Provide
import asyncio

from app.containers import Container
from app.analytics.collector import AnalyticsCollector

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/analytics")
@inject
async def websocket_endpoint(
    websocket: WebSocket,
    collector: AnalyticsCollector = Depends(Provide[Container.analytics_collector])
):
    logger.info("Analytics client trying to connect...")
    await collector.connect(websocket)
    logger.info("Analytics client connected successfully.")
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("Client disconnected (WebSocketDisconnect exception).")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the websocket endpoint: {e}", exc_info=True)
    finally:
        logger.warning("Websocket endpoint is closing. Disconnecting client.")
        collector.disconnect(websocket)
