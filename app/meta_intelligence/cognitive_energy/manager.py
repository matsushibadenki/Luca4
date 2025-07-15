# /app/meta_intelligence/cognitive_energy/manager.py
# title: 認知エネルギーマネージャー
# role: システム全体の「思考体力」をシミュレートし、管理するシングルトンクラス。

import time
import logging
import threading

logger = logging.getLogger(__name__)

class CognitiveEnergyManager:
    """
    システムの「思考体力」をシミュレートするシングルトンクラス。
    エネルギーは有限であり、思考によって消費され、アイドル時間で回復する。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_energy: float = 100.0, recovery_rate: float = 1.0):
        """
        CognitiveEnergyManagerを初期化します。
        シングルトンのため、初期化は一度しか実行されません。
        """
        if not hasattr(self, '_initialized'):
            self.max_energy = max_energy
            self.current_energy = max_energy
            self.recovery_rate = recovery_rate  # 1秒あたりの回復量
            self.last_update_time = time.time()
            self._initialized = True
            logger.info(f"CognitiveEnergyManager initialized with max_energy={self.max_energy}, recovery_rate={self.recovery_rate}")

    def _update_energy(self):
        """
        最後の更新からの経過時間に基づいてエネルギーを回復させる内部メソッド。
        """
        with self._lock:
            now = time.time()
            elapsed_time = now - self.last_update_time
            recovered_energy = elapsed_time * self.recovery_rate
            self.current_energy = min(self.max_energy, self.current_energy + recovered_energy)
            self.last_update_time = now

    def consume_energy(self, cost: float) -> bool:
        """
        指定されたコストのエネルギーを消費する。
        成功すればTrue、エネルギー不足で失敗すればFalseを返す。
        """
        self._update_energy()
        with self._lock:
            if self.current_energy >= cost:
                self.current_energy -= cost
                logger.info(f"Energy consumed: {cost}. Current energy: {self.current_energy:.2f}")
                return True
            else:
                logger.warning(f"Failed to consume energy. Cost: {cost}, Current energy: {self.current_energy:.2f}")
                return False

    def get_current_energy_level(self) -> float:
        """
        現在のエネルギーレベルを返す。
        内部で回復処理を行ってから最新の値を返す。
        """
        self._update_energy()
        return self.current_energy
    
    def _recover_energy(self):
        """
        IdleManagerから定期的に呼び出されるためのエネルギー回復メソッド。
        実質的には_update_energyのエイリアスとして機能する。
        """
        self._update_energy()
        logger.debug(f"Energy recovered. Current energy: {self.current_energy:.2f}")