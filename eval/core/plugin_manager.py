import importlib
import logging
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

class PluginManager:
    def __init__(self, plugin_configs: List[Dict]):
        self.plugin_configs = plugin_configs
        self.plugins = []
        self._load_plugins()

    def _load_plugins(self):
        # 插件路由映射表：每个维度对应一个独立的插件类
        plugin_map = {
            "wer": "eval.plugins.wer_plugin.WERPlugin",
            "llm_naturalness": "eval.plugins.llm_naturalness.LLMNaturalnessPlugin",
            "llm_intelligibility": "eval.plugins.llm_intelligibility.LLMIntelligibilityPlugin",
            "llm_language": "eval.plugins.llm_language.LLMLanguagePlugin",
            "llm_voice": "eval.plugins.llm_voice.LLMVoicePlugin",
            "llm_text_type": "eval.plugins.llm_text_type.LLMTextTypePlugin",
            "text_length": "eval.plugins.text_length_plugin.TextLengthPlugin",
            "dnsmos": "eval.plugins.dnsmos_plugin.DNSMOSPlugin",
            # 未来可扩展
            # "speaker_similarity": "eval.plugins.speaker_sim_plugin.SpeakerSimPlugin",
        }
        
        for p_conf in self.plugin_configs:
            p_name = p_conf.get("name")
            if not p_name:
                continue
                
            if p_name not in plugin_map:
                logger.warning(f"Plugin '{p_name}' not found in internal map. Skipping.")
                continue
                
            module_path, class_name = plugin_map[p_name].rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                plugin_class = getattr(module, class_name)
                # 动态实例化，传入专属的配置
                plugin_instance = plugin_class(config=p_conf.get("config", {}))
                self.plugins.append(plugin_instance)
                logger.info(f"Successfully loaded plugin: {p_name}")
            except Exception as e:
                logger.error(f"Failed to load plugin '{p_name}': {e}", exc_info=True)

    def execute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """顺序执行所有已加载的插件"""
        for plugin in self.plugins:
            logger.info(f"Running plugin: {plugin.name}")
            try:
                df = plugin.run(df)
            except Exception as e:
                # 错误兜底机制：任一插件崩溃均不影响大局
                logger.error(f"Plugin {plugin.name} execution failed: {e}", exc_info=True)
        return df