import logging
import pandas as pd
import yaml
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from eval.core.plugin_manager import PluginManager
from eval.core.fetch import fetch_samples_postgres, fetch_samples_bq
from eval.core.audio import download_audio
from eval.core.asr import transcribe_audio

logger = logging.getLogger(__name__)

class EvalPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.plugin_manager = PluginManager(self.config.get("plugins", []))
        
    def _process_audio_and_asr(self, idx: int, row: pd.Series) -> tuple:
        file_path_gcs = row.get("file_path")
        case_id = row.get("case_id", f"idx_{idx}")
        
        local_path_obj = None
        asr_result = None
        
        if file_path_gcs:
            # 尝试下载
            try:
                local_path_obj = download_audio(file_path_gcs, dest_name=f"{case_id}.mp3")
            except Exception as e:
                logger.error(f"Failed to download audio for {case_id}: {e}")
                
        if local_path_obj and local_path_obj.exists():
            # 下载成功，调用 ASR
            try:
                asr_result = transcribe_audio(local_path_obj)
            except Exception as e:
                logger.error(f"Failed to transcribe audio for {case_id}: {e}")
        else:
            logger.warning(f"Audio not found or download failed for {case_id}, skipping ASR.")

        return (
            idx,
            str(local_path_obj.resolve()) if local_path_obj else None,
            asr_result
        )

    def sample_data(self) -> pd.DataFrame:
        """
        从数据源抽样。从 Postgres 或 BigQuery 捞取数据，
        并下载音频，调用 Gemini ASR 转写文本，最终组装为 DataFrame。
        """
        sampling_cfg = self.config.get("sampling", {})
        sample_size = sampling_cfg.get("size", 500)
        source = sampling_cfg.get("source", "postgres") # 默认 postgres
        
        logger.info(f"Sampling up to {sample_size} records from {source}...")
        
        if source == "postgres":
            samples = fetch_samples_postgres(limit=sample_size, undownloaded_only=False)
        else:
            samples = fetch_samples_bq(limit=sample_size, undownloaded_only=False)

        if not samples:
            logger.warning("No samples fetched from database.")
            return pd.DataFrame()

        df = pd.DataFrame(samples)
        
        # 将 target_text 重命名为 text 适配后续逻辑，同时保留原有字段习惯
        if "target_text" in df.columns:
            df["text"] = df["target_text"]
        if "gen_product_id" in df.columns:
            df["case_id"] = df["gen_product_id"]
            
        # 补全语言标识（若数据库未直接提供，可默认为空或后续由插件补充）
        # if "language" not in df.columns:
        #    df["language"] = "unknown"

        # 下载音频并转写
        logger.info("Downloading audios and transcribing via Gemini ASR...")
        
        # 使用 ThreadPoolExecutor 并发处理下载和 ASR
        results_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for idx, row in df.iterrows():
                futures.append(executor.submit(self._process_audio_and_asr, idx, row))
                # 控制一下速率防限频，每次提交增加少许间隔
                time.sleep(0.1)
                
            for future in futures:
                try:
                    res_idx, audio_path, asr_text = future.result()
                    results_map[res_idx] = (audio_path, asr_text)
                except Exception as e:
                    logger.error(f"Error processing row: {e}")

        local_audio_paths = []
        asr_texts = []
        for idx in df.index:
            path, text = results_map.get(idx, (None, None))
            local_audio_paths.append(path)
            asr_texts.append(text)

        df["audio_path"] = local_audio_paths
        df["asr_text"] = asr_texts

        # 导出原始数据 CSV
        raw_export_path = Path(sampling_cfg.get("export_raw", "eval/output/raw_cases.csv"))
        raw_export_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_export_path, index=False)
        logger.info(f"Exported raw sampled data with ASR to {raw_export_path}")
        return df
        
    def run(self):
        """执行完整评测流水线"""
        logger.info("Starting evaluation pipeline...")
        df = self.sample_data()
        
        if df.empty:
            logger.error("Data sample is empty. Aborting pipeline.")
            return
            
        # 依次执行所有配置驱动的插件
        df = self.plugin_manager.execute_all(df)
        
        # 导出最终包含所有打标与指标的 CSV
        # 按用户要求优化输出字段：删除 language（保留 llm_language_type），将路径字段放到最后
        if "language" in df.columns:
            df = df.drop(columns=["language"])
            
        # 整理列顺序：提取非路径列和路径列
        path_cols = [col for col in df.columns if col in ["file_path", "audio_path"]]
        other_cols = [col for col in df.columns if col not in path_cols]
        
        # 将路径列放到最后
        df = df[other_cols + path_cols]
        
        final_export_path = Path(self.config.get("sampling", {}).get("export_final", "eval/output/final_results.csv"))
        final_export_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(final_export_path, index=False)
        logger.info(f"Pipeline finished. Final results exported to {final_export_path}")