import pandas as pd
import logging
from eval.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

class DNSMOSPlugin(BasePlugin):
    """
    DNSMOS（Deep Noise Suppression MOS）音质评估插件。
    此插件负责调用本地 ONNX 模型进行推理，并安全地合并结果。
    """
    name = "dnsmos"

    def __init__(self, config: dict = None):
        super().__init__(config)
        
        self.model_path = self.config.get("model_path", "eval/plugins/dnsmos/model.onnx")
        self.device = self.config.get("device", "cpu")
        self.model = None
        
        try:
            from eval.plugins.dnsmos.dnsmos import DNSMOSModel
            self.model = DNSMOSModel(model_path=self.model_path, device=self.device)
            logger.info(f"Successfully loaded DNSMOS model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load DNSMOS model from {self.model_path}. Is onnxruntime installed? Error: {e}")

    def _compute_dnsmos(self, row):
        # 预留给外部传递音频路径的字段，目前暂设为 audio_path
        audio_path = row.get("audio_path", "")
        
        if not audio_path or pd.isna(audio_path):
            return pd.Series({"dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovr": None, "dnsmos_std": None})
            
        if self.model is None:
            return pd.Series({"dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovr": None, "dnsmos_std": None})
            
        try:
            score = self.model.infer(audio_path)
            return pd.Series({
                "dnsmos_sig": score.get("dnsmos_sig"),
                "dnsmos_bak": score.get("dnsmos_bak"),
                "dnsmos_ovr": score.get("dnsmos_ovr"),
                "dnsmos_std": score.get("dnsmos_std")
            })
        except Exception as e:
            logger.error(f"DNSMOS compute error for case {row.get('case_id', 'unknown')}: {e}")
            return pd.Series({"dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovr": None, "dnsmos_std": None})

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if "audio_path" not in df.columns:
            logger.warning("Column 'audio_path' not found. DNSMOS computation will be skipped.")
            df["dnsmos_sig"] = None
            df["dnsmos_bak"] = None
            df["dnsmos_ovr"] = None
            df["dnsmos_std"] = None
            return df
            
        metrics = df.apply(self._compute_dnsmos, axis=1)
        
        df["dnsmos_sig"] = metrics["dnsmos_sig"]
        df["dnsmos_bak"] = metrics["dnsmos_bak"]
        df["dnsmos_ovr"] = metrics["dnsmos_ovr"]
        df["dnsmos_std"] = metrics["dnsmos_std"]
        
        return df