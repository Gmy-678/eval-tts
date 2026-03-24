import pandas as pd
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from eval.plugins.base import BasePlugin
from eval.core.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class BaseLLMPlugin(BasePlugin):
    """
    LLM 评测插件的基类。处理通用流程（调 LLM，解 JSON，追加列）
    子类需要实现 _get_prompt(row) 和提供 name。
    """
    name = "base_llm"

    def __init__(self, config: dict = None):
        super().__init__(config)
        model_name = self.config.get("model", "gemini-2.5-pro")
        max_retries = self.config.get("max_retries", 3)
        self.client = GeminiClient(model_name=model_name, max_retries=max_retries)
        self.prompt_template = self.config.get("prompt_template", "")

    def _get_prompt(self, row: pd.Series) -> str:
        """根据行数据生成 Prompt，子类可覆盖"""
        if not self.prompt_template:
            logger.warning(f"{self.name}: prompt_template is empty.")
            
        # 兼容处理：在 yaml 中我们的 prompt 模板不再传 language，或者传入
        try:
            return self.prompt_template.format(
                text=row.get("text", ""),
                language=row.get("language", "unknown")
            )
        except KeyError:
            return self.prompt_template.format(
                text=row.get("text", "")
            )

    def _annotate_row(self, row: pd.Series) -> dict:
        prompt = self._get_prompt(row)
        fallback = {"error": f"{self.name} annotation timeout or failed"}
        return self.client.generate_json(prompt, fallback=fallback)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # LLM 打标
        json_col_name = f"{self.name}_raw_json"
        
        results_map = {}
        
        def process(idx, row):
            return idx, self._annotate_row(row)
            
        # 使用 ThreadPoolExecutor 并发处理打标
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for idx, row in df.iterrows():
                futures.append(executor.submit(process, idx, row))
                time.sleep(0.05)  # 少量限速防并发过高
                
            for future in futures:
                try:
                    idx, res = future.result()
                    results_map[idx] = res
                except Exception as e:
                    logger.error(f"Error in {self.name} annotation: {e}")

        # 保证按照 df 的原始顺序填充
        results_list = [results_map.get(idx, {"error": "annotation failed"}) for idx in df.index]
        df[json_col_name] = results_list
        
        # 将 JSON 结果展开并追加到原表
        annotations_df = pd.json_normalize(df[json_col_name].tolist())
        annotations_df.index = df.index
        # 为展开的列增加前缀，防止冲突，如 llm_naturalness_score
        annotations_df = annotations_df.add_prefix(f"{self.name}_")
        
        # 用户要求：如果是 llm_language，删除 llm_language_language 字段，保留 llm_language_type
        if self.name == "llm_language" and "llm_language_language" in annotations_df.columns:
            annotations_df = annotations_df.drop(columns=["llm_language_language"])
            
        df = pd.concat([df.drop(columns=[json_col_name]), annotations_df], axis=1)
        return df