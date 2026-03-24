import re
import unicodedata
from typing import Optional

# =========================
# 配置
# =========================
USE_NEMO_ITN = True
# 数字归一：True=仅数值上下文替换（更严谨），False=全量替换（可能误伤 "the one who"）
USE_STRICT_NUMERIC_ONLY = True
_ITN_SUPPORTED = frozenset(
    {"en", "de", "es", "pt", "ru", "fr", "sv", "vi", "ar", "es_en"}
)

_ITN_CACHE = {}
try:
    from nemo_text_processing.inverse_text_normalization import InverseNormalizer
except ImportError:
    InverseNormalizer = None


# =========================
# 1. 控制 token 清理
# =========================
def remove_tts_tokens(text: str) -> str:
    """
    删除：
    [Happy#Happy:6]:
    [Happy#Happy:6;Anger:2]:
    """
    if not text:
        return ""

    return re.sub(r"\[[^\[\]]+:\d+(?:;[^\[\]]+:\d+)*\]:", "", text).strip()


def _has_chinese(text: str) -> bool:
    """检测文本是否含中文，用于自动触发中文归一化（繁简、数字）"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def tokenize_chinese_for_wer(text: str) -> str:
    """
    中文按字分词：在中文字符间插入空格，使 jiwer 的 WER 按字对齐。
    否则 ref 有空格（多 token）而 hyp 无空格（1 token）会导致 WER 虚高。
    """
    if not text or not _has_chinese(text):
        return text
    # 每个中文字符前后加空格，数字序列保持连续
    text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


# =========================
# 2. Unicode & 基础规范化
# =========================
def basic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    # 零宽字符
    text = re.sub(r"[\u200b-\u200d\ufeff\u2060]", "", text)

    # 空白统一
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# 3. Unicode 标点归一化（关键）
# =========================
def normalize_unicode_punct(text: str) -> str:
    """
    把所有“奇怪标点”拉回标准空间
    """
    text = text.replace("…", "...")
    text = text.replace("—", " ")
    text = text.replace("–", " ")

    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    return text


# =========================
# 4. 缩写展开（ITN 补强：ref/hyp 必须用同一套规则）
# =========================
_CONTRACTIONS = {
    "you're": "you are",
    "it's": "it is",
    "don't": "do not",
    "i'm": "i am",
    "they're": "they are",
    "we're": "we are",
    "can't": "can not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "didn't": "did not",
    "doesn't": "does not",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    "let's": "let us",
    "we've": "we have",
    "they've": "they have",
    "you've": "you have",
    "i've": "i have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "it'd": "it would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "it'll": "it will",
}


def expand_contractions(text: str) -> str:
    """
    缩写展开：you're → you are，统一 ref/hyp token 对齐。
    ⚠ ref 和 hyp 必须用同一套规则，否则引入误差。
    """
    tokens = text.split()
    return " ".join(_CONTRACTIONS.get(t, t) for t in tokens)


# =========================
# 5. 数字归一（ITN 补强：one → 1）
# =========================
# ⚠ 风险：不是所有 "one" 都是数字，如 "the one who left"。
# 更严谨做法：只在纯数字句子/数值上下文开启，或 strict_numeric_only=True。
_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_simple_numbers(text: str, strict_numeric_only: bool = False) -> str:
    """
    数字归一：one → 1，消除 ref: 1 / hyp: one 导致的 WER=100% 误判。
    strict_numeric_only=True 时，仅在相邻为数字或明显数值上下文时替换。
    """
    tokens = text.split()
    if strict_numeric_only:
        result = []
        for i, t in enumerate(tokens):
            if t not in _WORD_TO_DIGIT:
                result.append(t)
                continue
            # 前后有数字或纯数字 token 时视为数值上下文
            prev_num = i > 0 and (tokens[i - 1].isdigit() or tokens[i - 1] in _WORD_TO_DIGIT)
            next_num = i < len(tokens) - 1 and (tokens[i + 1].isdigit() or tokens[i + 1] in _WORD_TO_DIGIT)
            if prev_num or next_num:
                result.append(_WORD_TO_DIGIT[t])
            else:
                result.append(t)
        return " ".join(result)
    return " ".join(_WORD_TO_DIGIT.get(t, t) for t in tokens)


# =========================
# 5b. 中文繁简统一
# =========================
def normalize_chinese_traditional_to_simplified(text: str) -> str:
    """
    繁体 → 简体，统一 ref/hyp 映射空间。
    依赖 zhconv，未安装时原样返回。
    """
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except ImportError:
        return text


# =========================
# 5c. 中文数字归一化
# =========================
def normalize_chinese_numbers(text: str) -> str:
    """
    中文数字 → 阿拉伯数字，与 ASR 输出（阿拉伯数字）对齐。
    依赖 cn2an，未安装时用简单正则处理常见模式。
    """
    if not text or not re.search(r"[\u4e00-\u9fff]", text):
        return text

    try:
        import cn2an
        return cn2an.transform(text)  # 默认 cn2an：中文数字 → 阿拉伯数字
    except (ImportError, Exception):
        pass

    # 简单正则：常见单位 万、千、百、亿（cn2an 未安装或转换失败时回退）
    # 支持系数：阿拉伯数字 或 中文 一二三四五六七八九十
    _cn_coef = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

    def _mul(m, mult):
        g = m.group(1)
        if not g:
            n = 1
        elif g in _cn_coef:
            n = _cn_coef[g]
        elif g.isdigit():
            n = int(g)
        else:
            n = 1
        return str(n * mult)

    _pat = r"([一二三四五六七八九十零]|\d+)?"
    text = re.sub(_pat + "千万", lambda m: _mul(m, 10000000), text)
    text = re.sub(_pat + "百万", lambda m: _mul(m, 1000000), text)
    text = re.sub(_pat + "十万", lambda m: _mul(m, 100000), text)
    text = re.sub(_pat + "万", lambda m: _mul(m, 10000), text)
    text = re.sub(_pat + "千", lambda m: _mul(m, 1000), text)
    text = re.sub(_pat + "百", lambda m: _mul(m, 100), text)
    text = re.sub(_pat + "亿", lambda m: _mul(m, 100000000), text)

    return text


# =========================
# 6. 标点 → 空格（核心策略）
# =========================
def punctuation_to_space(text: str) -> str:
    """
    所有标点转空格（保留 '）
    """
    text = re.sub(r"[^\w\s\u4e00-\u9fff']", " ", text)
    return text


# =========================
# 7. 防爆机制
# =========================
def guard_explosion(before: str, after: str, ratio: float = 3.0) -> str:
    if not before.strip():
        return after

    b = before.split()
    a = after.split()

    if len(a) > ratio * max(1, len(b)):
        return before

    return after


# =========================
# 8. ITN（仅英文 + 防爆）
# =========================
def _get_itn(lang: str):
    if lang not in _ITN_CACHE:
        _ITN_CACHE[lang] = InverseNormalizer(
            input_case="lower_cased",
            lang=lang,
        )
    return _ITN_CACHE[lang]


def safe_itn(text: str, lang: str) -> str:
    if not (USE_NEMO_ITN and InverseNormalizer and lang in _ITN_SUPPORTED):
        return text

    try:
        itn = _get_itn(lang)
        out = itn.inverse_normalize(text, verbose=False)

        if not out:
            return text

        out = re.sub(r"\s+", " ", out).strip()

        return guard_explosion(text, out)

    except Exception:
        return text


# =========================
# 9. 主函数（最终用于 WER）
# =========================
def normalize_for_wer(text: str, lang: Optional[str] = "en") -> str:
    if not text or not isinstance(text, str):
        return ""

    lang = (lang or "en").lower()

    # 1️⃣ 控制 token
    text = remove_tts_tokens(text)

    # 2️⃣ 基础 normalize
    text = basic_normalize(text)

    if not text:
        return ""

    # 2b️⃣ 中文繁简统一（basic_normalize 之后、lowercase 之前）
    # lang=zh 或 文本含中文 时均触发，确保 ASR 与 target 同映射空间
    if lang.startswith("zh") or _has_chinese(text):
        text = normalize_chinese_traditional_to_simplified(text)

    # 3️⃣ Unicode 标点统一
    text = normalize_unicode_punct(text)

    # 4️⃣ lowercase（ITN 前做）
    text = text.lower()

    # 5️⃣ 缩写展开 + 数字归一（ref/hyp 同一套规则，坐标系统一）
    text = expand_contractions(text)
    text = normalize_simple_numbers(text, strict_numeric_only=USE_STRICT_NUMERIC_ONLY)

    # 5b️⃣ 中文数字归一化（与 ASR 阿拉伯数字输出对齐）
    if lang.startswith("zh") or _has_chinese(text):
        text = normalize_chinese_numbers(text)

    # 6️⃣ ITN（仅英文）
    if lang.startswith("en"):
        new_text = safe_itn(text, lang)
        text = guard_explosion(text, new_text)

    # 7️⃣ 标点 → 空格（关键）
    text = punctuation_to_space(text)

    # 8️⃣ 空白清理
    text = re.sub(r"\s+", " ", text)

    # 9️⃣ 中文按字分词（使 jiwer WER 按字对齐，避免 ref 有空格 hyp 无空格导致 WER 虚高）
    if _has_chinese(text):
        text = tokenize_chinese_for_wer(text)

    return text.strip()


# =========================
# 10. 轻量预处理（供 wer.py before_tn / already_normalized 用）
# =========================
def preprocess_for_wer(text: str, lang: Optional[str] = "en") -> str:
    """
    WER 前轻量预处理：不跑 ITN，仅基础规范化 + 缩写展开 + 数字归一 + 标点→空格。
    供 wer.py 的 before_tn、already_normalized 路径使用。
    ref/hyp 必须用同一套规则。
    """
    if not text or not isinstance(text, str):
        return ""

    lang = (lang or "en").lower()

    text = remove_tts_tokens(text)
    text = basic_normalize(text)
    if not text:
        return ""

    if lang.startswith("zh") or _has_chinese(text):
        text = normalize_chinese_traditional_to_simplified(text)

    text = normalize_unicode_punct(text)
    text = text.lower()
    text = expand_contractions(text)
    text = normalize_simple_numbers(text, strict_numeric_only=USE_STRICT_NUMERIC_ONLY)

    if lang.startswith("zh") or _has_chinese(text):
        text = normalize_chinese_numbers(text)

    text = punctuation_to_space(text)
    text = re.sub(r"\s+", " ", text)

    if _has_chinese(text):
        text = tokenize_chinese_for_wer(text)

    return text.strip()


# =========================
# 11. Debug（强烈建议）
# =========================
def debug_compare(ref: str, hyp: str):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    print("REF:", ref)
    print("HYP:", hyp)
    print("REF TOKENS:", len(ref_tokens))
    print("HYP TOKENS:", len(hyp_tokens))
    print("RATIO:", len(hyp_tokens) / max(1, len(ref_tokens)))