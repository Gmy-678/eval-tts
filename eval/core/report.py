import logging
import pandas as pd
import requests
import json
from datetime import date

logger = logging.getLogger(__name__)

def generate_report(df: pd.DataFrame) -> dict:
    """
    根据评测结果的 DataFrame 统计关键指标。
    """
    if df.empty:
        return {"total_samples": 0}

    total_samples = len(df)
    
    # 指标计算：WER/CER 的均值
    avg_wer = df["wer"].dropna().mean() if "wer" in df.columns else 0.0
    avg_cer = df["cer"].dropna().mean() if "cer" in df.columns else 0.0
    
    # 指标计算：DNSMOS 的均值
    avg_dnsmos_ovr = df["dnsmos_ovr"].dropna().mean() if "dnsmos_ovr" in df.columns else 0.0
    
    # 分类统计（如果存在 llm_language_type 等）
    lang_dist = {}
    if "llm_language_type" in df.columns:
        lang_dist = df["llm_language_type"].dropna().value_counts().to_dict()
        
    return {
        "date": date.today().strftime("%Y-%m-%d"),
        "total_samples": total_samples,
        "avg_wer": round(float(avg_wer), 4) if not pd.isna(avg_wer) else 0.0,
        "avg_cer": round(float(avg_cer), 4) if not pd.isna(avg_cer) else 0.0,
        "avg_dnsmos_ovr": round(float(avg_dnsmos_ovr), 2) if not pd.isna(avg_dnsmos_ovr) else 0.0,
        "language_distribution": lang_dist
    }

def send_webhook(report_data: dict, webhook_url: str):
    """
    将统计结果推送到飞书 (Lark) Webhook。
    """
    if not webhook_url or "YOUR_WEBHOOK_URL_HERE" in webhook_url:
        logger.warning("Webhook URL 未配置，跳过推送。")
        return

    # 飞书格式：msg_type 为 post (也可以选 interactive，这里用简单的 post 或 text)
    # 对于飞书，发送 Markdown 或者结构化信息建议用 post
    # https://open.larksuite.com/document/client-docs/bot-v3/add-custom-bot#%E6%94%AF%E6%8C%81%E5%8F%91%E9%80%81%E7%9A%84%E6%B6%88%E6%81%AF%E8%AF%B4%E6%98%8E
    
    # 构建飞书富文本(post)
    content_lines = [
        [{"tag": "text", "text": f"📅 日期: {report_data.get('date')}"}],
        [{"tag": "text", "text": f"📊 样本总数: {report_data.get('total_samples', 0)}"}],
        [{"tag": "text", "text": f"❌ 平均 WER (词错误率): {report_data.get('avg_wer', 0.0):.2%}"}],
        [{"tag": "text", "text": f"❌ 平均 CER (字错误率): {report_data.get('avg_cer', 0.0):.2%}"}],
        [{"tag": "text", "text": f"🔊 平均 DNSMOS (音质得分): {report_data.get('avg_dnsmos_ovr', 0.0)}"}],
    ]
    
    lang_dist = report_data.get("language_distribution", {})
    if lang_dist:
        content_lines.append([{"tag": "text", "text": "\n🌍 语种分布:"}])
        for lang, count in lang_dist.items():
            content_lines.append([{"tag": "text", "text": f"- {lang}: {count} ({count/max(report_data.get('total_samples', 1), 1):.1%})"}])

    gcs_path = report_data.get("gcs_path")
    if gcs_path:
        content_lines.append([{"tag": "text", "text": f"\n\n🔗 详情结果 CSV 请至 GCS 查看：\n{gcs_path}"}])

    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": "TTS 每日自动评测报告",
                    "content": content_lines
                }
            }
        }
    }

    try:
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10
        )
        if response.status_code == 200:
            logger.info("Webhook report sent successfully.")
        else:
            logger.error(f"Failed to send webhook report. HTTP {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error while sending webhook: {e}")
