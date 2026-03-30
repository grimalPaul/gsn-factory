from typing import List

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

# from src.models.score_pipeline import ScoreModule
from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def launch_evaluation(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("test")
    prompts_to_generate = datamodule.get_prompt_list()
    OmegaConf.set_struct(cfg.evaluation, False)
    cfg.evaluation["prompts_to_generate"] = prompts_to_generate
    OmegaConf.set_struct(cfg.evaluation, True)
    log.info(f"instantiating pipeline <{cfg.evaluation._target_}>")
    pipeline: LightningModule = hydra.utils.instantiate(cfg.evaluation)

    if not pipeline.all_is_processed:
        log.info(
            "Warning all the prompts of the datasets are not generated yet. Stopping the evaluation."
        )
        return None, None

    if "load_only_results" in cfg and cfg.load_only_results:
        pipeline.save_scores()
        return None, None

    log.info("Instantiate callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        inference_mode=False,
        callbacks=callbacks,
        # devices=find_usable_cuda_devices(1),
    )

    log.info("Starting evaluation!")
    trainer.test(datamodule=datamodule, model=pipeline)
    return None, None


@hydra.main(version_base=None, config_path="../configs", config_name="evaluation.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for test.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    log.info("Initialisation")
    launch_evaluation(cfg)


if __name__ == "__main__":
    main()
