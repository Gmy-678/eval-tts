import pandas as pd
import logging
from eval.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

class TextLengthPlugin(BasePlugin):
    name = "text_length"

    def _categorize_length(self, length: int) -> str:
        if length < 10:
            return "short"
        elif length <= 50:
            return "medium"
        else:
            return "long"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            logger.warning("Column 'text' not found. text_length will be empty.")
            df["text_length"] = None
            df["text_length_category"] = None
            return df
            
        df["text_length"] = df["text"].astype(str).apply(len)
        df["text_length_category"] = df["text_length"].apply(self._categorize_length)
        return df
