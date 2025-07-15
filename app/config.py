# /app/config.py
# title: アプリケーション設定
# role: アプリケーション全体で使用される設定値を一元管理する。

import os
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()

class Config:
    # --- LLMバックエンド設定 ---
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama") # 'ollama' または 'llama_cpp'
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    LAMA_CPP_MODEL_PATH: str = os.getenv("LAMA_CPP_MODEL_PATH", "/path/to/your/model.gguf") # llama.cppモデルのGGUFパス

    # LLM関連の設定 (OllamaとLlama.cppで共通のキーを持つ)
    GENERATION_LLM_SETTINGS: Dict[str, Any] = {
        "model": "gemma3:latest", # Ollamaの場合のモデル名
        "temperature": 0.7,
        "n_ctx": 2048, # Llama.cppの場合のコンテキスト長
        "n_batch": 512, # Llama.cppの場合のバッチサイズ
        "n_gpu_layers": -1, # 追加: Llama.cppの場合にGPUにオフロードする層の数 (-1は可能な限り全て)
    }
    VERIFIER_LLM_SETTINGS: Dict[str, Any] = {
        "model": "gemma3:latest", # Ollamaの場合のモデル名
        "temperature": 0.4,
        "n_ctx": 2048, # Llama.cppの場合のコンテキスト長
        "n_batch": 512, # Llama.cppの場合のバッチサイズ
        "n_gpu_layers": 0, # 追加: Llama.cppの場合にGPUにオフロードする層の数 (0はCPUのみ)
    }
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"

    # ファイルパス関連: 環境変数からの読み込みを可能にする
    KNOWLEDGE_BASE_SOURCE: str = os.getenv("KNOWLEDGE_BASE_SOURCE", "data/documents/initial_facts.txt")
    KNOWLEDGE_GRAPH_STORAGE_PATH: str = os.getenv("KNOWLEDGE_GRAPH_STORAGE_PATH", "memory/knowledge_graph.json")
    MEMORY_LOG_FILE_PATH: str = os.getenv("MEMORY_LOG_FILE_PATH", "memory/session_memory.jsonl")

    # パイプラインごとの設定
    PIPELINE_SETTINGS: Dict[str, Dict[str, int]] = {
        "speculative": {
            "num_drafts": 3
        },
        "internal_dialogue": {
            "max_turns": 5
        },
        "cognitive_loop": {
            "max_iterations": 3
        }
    }

    # アイドル時間と自律思考の実行間隔（秒）
    IDLE_EVOLUTION_TRIGGER_SECONDS: int = 30
    AUTONOMOUS_CYCLE_INTERVAL_SECONDS: int = 60
    CONSOLIDATION_CYCLE_INTERVAL_SECONDS: int = 300
    WISDOM_SYNTHESIS_INTERVAL_SECONDS: int = 600
    SIMULATION_CYCLE_INTERVAL_SECONDS: int = 600
    MICRO_LLM_CREATION_INTERVAL_SECONDS: int = 7200
    BENCHMARK_INTERVAL_SECONDS: int = 3600 # 1時間に1回ベンチマークを実行

    # 強化学習エージェント（PPO）の設定
    RL_AGENT_SETTINGS: Dict[str, Dict[str, Any]] = {
        "ppo": {
            "lr_actor": 0.0003,
            "lr_critic": 0.001,
            "gamma": 0.99,
            "K_epochs": 80,
            "eps_clip": 0.2,
            "max_ep_len": 20,
            "update_timestep_factor": 4
        }
    }

    # 自律思考エージェントが探求する初期トピックのリスト
    AUTONOMOUS_RESEARCH_TOPICS: List[str] = [
        "最新のAI技術トレンド",
        "持続可能なエネルギー源",
        "宇宙探査の進捗",
        "健康的な食事と運動",
        "世界の経済動向",
        "核融合エネルギー"
    ]

    # QuantumInspiredPipelineで使用するペルソナのリスト
    QUANTUM_PERSONAS: List[Dict[str, str]] = [
        {"name": "楽観的な未来学者", "persona": "あなたは未来の可能性を信じる楽観的な未来学者です。"},
        {"name": "懐疑的なリスクアナリスト", "persona": "あなたは何事にも潜むリスクを冷静に分析する懐疑的なリスクアナリストです。"},
        {"name": "共感的な倫理学者", "persona": "あなたは技術が人間に与える影響を深く考える、共感力の高い倫理学者です。"},
        {"name": "実践的なエンジニア", "persona": "あなたは理論よりも実践的な解決策を重視する現実的なエンジニアです。"},
    ]

    # 価値観の初期設定
    INITIAL_CORE_VALUES: Dict[str, float] = {
        "Helpfulness": 0.8,
        "Harmlessness": 0.9,
        "Honesty": 0.85,
        "Empathy": 0.7,
    }

settings = Config()