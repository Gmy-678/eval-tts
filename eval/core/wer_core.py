"""WER/CER 字准率计算（jiwer），用于 ASR 转写结果与 target_text 对齐"""
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from jiwer import cer, process_words, wer

from eval.core.normalize import normalize_for_wer, preprocess_for_wer

try:
    from eval.core.config import WER_LANG as _DEFAULT_WER_LANG
except ImportError:
    _DEFAULT_WER_LANG = "en"


def _is_chinese(s: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in s)


def _is_english(s: str) -> bool:
    return any(c.isalpha() and ord(c) < 128 for c in s)


def _has_repeat(s: str) -> bool:
    # 连续3个以上重复字符
    return bool(re.search(r"(.)\1{2,}", s))


def _edit_distance(a: str, b: str) -> int:
    # 简单DP（token级已对齐，这里用于词级判断）
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def analyze_error(ref: str, hyp: str) -> str:
    """
    输出简要错误原因
    """
    try:
        if ref == hyp:
            return "完全正确"

        out = process_words(ref, hyp)

        subs, ins, dels = [], [], []

        ref_tokens = out.references[0]
        hyp_tokens = out.hypotheses[0]
        for op in out.alignments[0]:
            if op.type == "substitute":
                r = " ".join(ref_tokens[op.ref_start_idx:op.ref_end_idx])
                h = " ".join(hyp_tokens[op.hyp_start_idx:op.hyp_end_idx])
                subs.append((r, h))
            elif op.type == "insert":
                h = " ".join(hyp_tokens[op.hyp_start_idx:op.hyp_end_idx])
                ins.append(h)
            elif op.type == "delete":
                r = " ".join(ref_tokens[op.ref_start_idx:op.ref_end_idx])
                dels.append(r)

        parts = []

        # ===== 替换分析（重点）=====
        for r, h in subs[:2]:
            tag = ""

            # 1. 中英文不一致
            if _is_chinese(r) != _is_chinese(h):
                tag = "（语言不一致）"

            # 2. 重复字符（ASR噪声）
            elif _has_repeat(r) or _has_repeat(h):
                tag = "（重复字符）"

            # 3. 近似拼写（编辑距离小）
            elif _edit_distance(r, h) <= max(1, min(len(r), len(h)) * 0.3):
                tag = "（拼写错误）"

            parts.append(f"{r} → {h}{tag}")

        # ===== 插入 / 删除 =====
        if ins:
            parts.append(f"多出: {ins[0]}")
        if dels:
            parts.append(f"缺失: {dels[0]}")

        return "；".join(parts) if parts else "轻微差异"

    except Exception:
        return "分析失败"


def check_token_mismatch(
    reference: str,
    hypothesis: str,
    res: dict,
    label: str = "",
) -> None:
    """
    若 ref/hyp token 数比例异常（ratio > 2 或 < 0.5），打印便于排查 badcase。
    在跑批时于 evaluate_stt 外层调用。
    """
    ref_clean = res.get("cleaned_ref", "")
    hyp_clean = res.get("cleaned_hyp", "")
    ref_len = len(ref_clean.split())
    hyp_len = len(hyp_clean.split())
    ratio = ref_len / hyp_len if hyp_len > 0 else (float("inf") if ref_len > 0 else 1.0)
    if ratio > 2 or ratio < 0.5:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}==== TOKEN MISMATCH ====")
        print("REF_RAW:", repr(reference))
        print("HYP_RAW:", repr(hypothesis))
        print("REF_CLEAN:", repr(ref_clean))
        print("HYP_CLEAN:", repr(hyp_clean))
        print("REF_LEN:", ref_len, "HYP_LEN:", hyp_len)
        print("RATIO:", ratio)


def evaluate_stt(
    reference: str,
    hypothesis: str,
    lang: Optional[str] = None,
    already_normalized: bool = False,
    before_tn: bool = False,
) -> dict:
    """
    计算 WER 和 CER。

    WER/CER 由 cleaned_ref 与 cleaned_hyp 计算：ref/hyp 经 normalize 后得到 cleaned_ref、cleaned_hyp，
    再传入 jiwer.wer()、jiwer.cer()。返回值限制在 [0, 1]。

    reference: 参考文本（target_text 或 target_tn）
    hypothesis: ASR 转写结果（asr_text 或 asr_tn）
    lang: 文本规范化语言，默认从 config.WER_LANG 读取
    already_normalized: 若 True，认为 ref/hyp 已是 normalize_for_wer 结果，仅做 preprocess_for_wer 后比对
    before_tn: 若 True，不做 normalize_for_wer，仅做 preprocess_for_wer（用于 TN 前对比：target_text vs asr_text）
    """
    lang = lang or _DEFAULT_WER_LANG
    if before_tn:
        ref_clean = preprocess_for_wer(reference, lang)
        hyp_clean = preprocess_for_wer(hypothesis, lang)
    elif already_normalized:
        ref_clean = preprocess_for_wer(reference, lang)
        hyp_clean = preprocess_for_wer(hypothesis, lang)
    else:
        ref_clean = normalize_for_wer(reference, lang)
        hyp_clean = normalize_for_wer(hypothesis, lang)

    # 边界：jiwer 在 ref/hyp 为空时返回 int 或异常值，统一处理并保证返回 [0,1] 的 float
    # 仅将真正空字符串视为空，不用 strip()，避免把 "   " 等 preprocess bug 误判为空
    if ref_clean == "":
        logger.warning("Empty ref after preprocess")
        wer_val = 1.0 if hyp_clean else 0.0
        cer_val = 1.0 if hyp_clean else 0.0
    elif hyp_clean == "":
        wer_val = 1.0
        cer_val = 1.0
    else:
        error_rate = wer(ref_clean, hyp_clean)
        char_error_rate = cer(ref_clean, hyp_clean)
        wer_val = min(1.0, max(0.0, float(error_rate)))
        cer_val = min(1.0, max(0.0, float(char_error_rate)))

    return {
        "wer": wer_val,
        "cer": cer_val,
        "cleaned_ref": ref_clean,
        "cleaned_hyp": hyp_clean,
    }


def evaluate_samples(
    samples: list,
    lang: Optional[str] = None,
    use_tn: bool = False,
) -> list:
    """
    批量 WER/CER 评估。
    use_tn=True 时用 target_tn vs asr_tn（已规范化），否则用 target_text vs asr_text。
    """
    lang = lang or _DEFAULT_WER_LANG
    for item in samples:
        if use_tn and "target_tn" in item and "asr_tn" in item:
            ref = item.get("target_tn", "")
            hyp = item.get("asr_tn", "")
            res = evaluate_stt(ref, hyp, lang=lang, already_normalized=True)
        else:
            ref = item.get("target_text", "")
            hyp = item.get("asr_text", "")
            res = evaluate_stt(ref, hyp, lang=lang)
        check_token_mismatch(ref, hyp, res, label=item.get("gen_product_id", ""))
        item["wer"] = res["wer"]
        item["cer"] = res["cer"]
        item["cleaned_ref"] = res.get("cleaned_ref", "")
        item["cleaned_hyp"] = res.get("cleaned_hyp", "")
        item["error_summary"] = analyze_error(res.get("cleaned_ref", ""), res.get("cleaned_hyp", ""))
    return samples


if __name__ == "__main__":
    # reference = "嘿,如果我说实话,你不赶我走,我就告诉你,为了泡妞,你懂吗?师傅果然目光如炬,我这点心思瞒不过您老人家。为了泡,为了长生不老。哦,这难道就是传说中丹药的味道。100多年,我靠,师傅,吃完的竹签不要乱扔,拜托你珍惜我的劳动成果,好不好?师傅,吃个大鸡腿,师傅,这大羊腿好。来来师傅,这牛肉也不错。"
    # hypothesis = "哈,如果我说实话,你不赶我走,我就告诉你,为了泡妞,你懂吗?师傅果然目光如炬,我这点心思瞒不过您老人家。为了泡,为了长生不老。哦,这难道就是传说中丹药的味道。100多年,我靠,师傅,吃完的竹签不要乱扔,拜托你珍惜我的劳动成果,好不好?师傅,吃个大鸡腿,师傅,这大羊腿好。来来师傅,这牛肉也不错。"
    
    reference = "tune into the night"
    hypothesis = "tune into the night"

    print(evaluate_stt(reference, hypothesis, before_tn=True))