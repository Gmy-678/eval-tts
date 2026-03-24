from eval.plugins.base_llm import BaseLLMPlugin

class LLMTextTypePlugin(BaseLLMPlugin):
    """
    打标文本类型:
    plain text（普通句子）, number（数字）, time/date（时间）, currency（金额）, proper noun（专有名词）, abbreviation（缩写）
    """
    name = "llm_text_type"
