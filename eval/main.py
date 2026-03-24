import os
import sys
import logging
from pathlib import Path

# Ensure the root of the project is in the PYTHONPATH so imports work correctly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.core.pipeline import EvalPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    pipeline = EvalPipeline(config_path)
    pipeline.run()