import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from ta_src.utils.quiet import silence_third_party

silence_third_party()

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    from ta_src.anonymization.comfyui_client import ComfyUIClient
    from ta_src.anonymization.reference_crop_cache import ReferenceCropCache
    from ta_src.pipeline.main_pipeline import Pipeline

    # Reuse Pipeline._seed_gallery via a minimal Pipeline construction. Easier:
    # call _load_stages, which builds gallery + crop_cache + anonymizer and
    # invokes prewarm_references at construction time.
    cfg.pipeline.run_anonymization = True
    cfg.pipeline.run_tracking = True
    cfg.pipeline.run_pose = False  # not needed for prewarm
    Pipeline(cfg)  # constructor runs prewarm; no .run() = no video processing
    log.info("Prewarm-only complete.")


if __name__ == "__main__":
    main()
