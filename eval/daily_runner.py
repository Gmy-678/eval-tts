import os
import sys
import json
import logging
import shutil
import yaml
from pathlib import Path
from datetime import date

# Ensure the root of the project is in the PYTHONPATH so imports work correctly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.core.pipeline import EvalPipeline
from eval.core.gcs_sink import upload_to_gcs
from eval.core.report import generate_report, send_webhook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def archive_results(config: dict):
    """
    压缩当天的原始评测数据和最终评测数据到归档目录。
    """
    today_str = date.today().strftime("%Y-%m-%d")
    
    # 确定路径，使用绝对路径避免 cron 运行时的相对路径问题
    sampling_cfg = config.get("sampling", {})
    raw_path = project_root / sampling_cfg.get("export_raw", "eval/output/raw_cases.csv")
    final_path = project_root / sampling_cfg.get("export_final", "eval/output/final_results.csv")
    
    daily_cfg = config.get("daily_job", {})
    archive_dir = project_root / daily_cfg.get("archive_dir", "eval/backup")
    
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        
    archive_base_name = f"{today_str}_eval_results"
    archive_path = archive_dir / archive_base_name
    
    # 将文件临时拷贝到压缩准备目录
    temp_dir = archive_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if raw_path.exists():
        shutil.copy2(raw_path, temp_dir / raw_path.name)
    if final_path.exists():
        shutil.copy2(final_path, temp_dir / final_path.name)
        
    try:
        shutil.make_archive(str(archive_path), 'zip', root_dir=temp_dir)
        logger.info(f"Archived results to {archive_path}.zip")
    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)

def main():
    # 切换工作目录到项目根目录，以保证各类相对路径不随 cron 运行位置改变而跑偏
    os.chdir(project_root)
    
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    # 读取配置
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return
        
    daily_cfg = config.get("daily_job", {})
    gcs_prefix = daily_cfg.get("gcs_prefix", "eval_results")
    webhook_url = daily_cfg.get("webhook_url", "")
    # 我们恢复飞书发送进行测试，或者你可以在这里关掉
    
    logger.info("=== 1. Running Daily Evaluation Pipeline ===")
    pipeline = EvalPipeline(config_path)
    df = pipeline.run()
    
    if df is None or df.empty:
        logger.warning("Pipeline returned no data. Ending daily job.")
        return
        
    logger.info("=== 2. Uploading Results to GCS ===")
    today_str = date.today().strftime("%Y-%m-%d")
    final_path = project_root / config.get("sampling", {}).get("export_final", "eval/output/final_results.csv")
    
    gcs_path = ""
    if final_path.exists():
        # Rename the file temporarily to use today's date, so the GCS blob has a nice name
        temp_gcs_upload_path = final_path.parent / f"{today_str}_results.csv"
        shutil.copy2(final_path, temp_gcs_upload_path)
        
        gcs_path = upload_to_gcs(str(temp_gcs_upload_path), gcs_prefix)
        # remove temporary file after upload
        temp_gcs_upload_path.unlink()
    
    logger.info("=== 3. Generating and Sending Report ===")
    report_data = generate_report(df)
    report_data['gcs_path'] = gcs_path
    print("======= 本次报告生成预览 =======")
    print(json.dumps(report_data, indent=2, ensure_ascii=False))
    print("================================")
    send_webhook(report_data, webhook_url)
    
    logger.info("=== 4. Archiving Results ===")
    archive_results(config)
    
    logger.info("=== Daily Evaluation Automation Finished ===")

if __name__ == "__main__":
    main()
