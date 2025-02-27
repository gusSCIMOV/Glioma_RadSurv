
import os
import logging
# internal imports
from preprocessing.preprocessing_core import MRI_preprocessing
from utils.config_loader import * 
from utils.logger_setup import *

#
#os.environ["PYTHONUTF8"] = "1"

# Logging setup
setup_logging(log_file="DicoConversion.log", log_level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(" Starting radiomics risk score MRI pipeline...")

# project PATH
project_path = os.path.dirname(os.path.abspath(__file__))

def main():
    logger.info(" Loading modules configuration: YAML keys are directly mapped as attributes in Config2Struct object: see config.confi_loader.py ...")
    # Load all configurations (main.yaml,preprocessing.yaml,radiomics.yaml, surival.yaml )
    configs = load_all_configs(project_path + "/config/main_config.yaml")

   # Pre-processing (enabled config)
    MRI_preprocessing(configs, project_path)

    print(f"Pipeline completed with output at ")


if __name__ == "__main__":
    main()
