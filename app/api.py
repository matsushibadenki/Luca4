# /app/api.py
# title: FastAPI APIエンドポイント
# role: 外部システムがLuca3と対話するためのRESTful APIを提供する。

import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from dependency_injector.wiring import inject, Provide
from pydantic import BaseModel, Field
from typing import Dict, Any

from app.containers import Container
from app.engine import MetaIntelligenceEngine
from app.agents.orchestration_agent import OrchestrationAgent
from app.models import MasterAgentResponse, OrchestrationDecision

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    query: str = Field(..., description="ユーザーからの入力テキスト")
    session_id: str | None = Field(None, description="（オプション）対話セッションを継続するためのID")

class ChatResponse(BaseModel):
    final_answer: str
    self_criticism: str
    potential_problems: str
    retrieved_info: str
    session_id: str | None = Field(None, description="応答に対応するセッションID")

class HealthStatus(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"

router = APIRouter()

@router.get("/status", response_model=HealthStatus, summary="APIの動作状況を確認")
async def get_status():
    return HealthStatus()

@router.post("/chat", response_model=ChatResponse, summary="AIとの対話")
@inject
async def chat_with_agent(
    request: ChatRequest,
    engine: MetaIntelligenceEngine = Depends(Provide[Container.engine]),
    orchestration_agent: OrchestrationAgent = Depends(Provide[Container.orchestration_agent]),
):
    try:
        logger.info(f"Received chat request with query: '{request.query}'")
        
        logger.info("オーケストレーションエージェントによるモード選択を開始します...")
        orchestration_decision: OrchestrationDecision = orchestration_agent.invoke({"query": request.query})
        logger.info(f"選択されたモード: {orchestration_decision.get('chosen_mode')}, 理由: {orchestration_decision.get('reason')}")

        response: MasterAgentResponse = await engine.arun(
            query=request.query, 
            orchestration_decision=orchestration_decision
        )

        return ChatResponse(
            final_answer=response["final_answer"],
            self_criticism=response["self_criticism"],
            potential_problems=response["potential_problems"],
            retrieved_info=response["retrieved_info"],
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
