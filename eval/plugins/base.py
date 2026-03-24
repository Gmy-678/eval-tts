import pandas as pd
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class BasePlugin(ABC):
    name: str = "base"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        接收 DataFrame，追加评测维度列后返回
        """
        pass