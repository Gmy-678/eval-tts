import pandas as pd
import logging
from eval.plugins.base import BasePlugin
from eval.core.wer_core import evaluate_stt, analyze_error

logger = logging.getLogger(__name__)

class WERPlugin(BasePlugin):
    name = "wer"

    def _compute_wer_cer(self, row):
        target = str(row.get("text", ""))
        asr = str(row.get("asr_text", ""))
        lang = str(row.get("language", ""))
        if not lang or lang.lower() == 'unknown':
            lang = None
        
        # 兜底机制：缺失则跳过，nan 的处理
        if not target or target.lower() == 'nan' or not asr or asr.lower() == 'nan':
            return pd.Series({"wer": None, "cer": None, "cleaned_ref": None, "cleaned_hyp": None, "error_summary": None})
            
        try:
            # 兼容 badcase-wer 的处理逻辑：通过 normalize_for_wer 预处理，然后计算 WER/CER
            res = evaluate_stt(reference=target, hypothesis=asr, lang=lang)
            err_summary = analyze_error(res.get("cleaned_ref", ""), res.get("cleaned_hyp", ""))
            
            return pd.Series({
                "wer": res.get("wer"), 
                "cer": res.get("cer"),
                "cleaned_ref": res.get("cleaned_ref"),
                "cleaned_hyp": res.get("cleaned_hyp"),
                "error_summary": err_summary
            })
        except Exception as e:
            logger.error(f"WER/CER compute error for case {row.get('case_id')}: {e}")
            return pd.Series({"wer": None, "cer": None, "cleaned_ref": None, "cleaned_hyp": None, "error_summary": None})

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if "asr_text" not in df.columns:
            logger.warning("Column 'asr_text' not found. WER/CER will be empty.")
            df["wer"] = None
            df["cer"] = None
            return df
            
        metrics = df.apply(self._compute_wer_cer, axis=1)
        df["wer"] = metrics["wer"]
        df["cer"] = metrics["cer"]
        df["cleaned_ref"] = metrics["cleaned_ref"]
        df["cleaned_hyp"] = metrics["cleaned_hyp"]
        df["error_summary"] = metrics["error_summary"]
        return df