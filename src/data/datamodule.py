import logging
from pathlib import Path

import lightning as L
import numpy as np
from datasets import DatasetDict, disable_caching, load_dataset, load_from_disk
from torch.utils.data import DataLoader

disable_caching()
logger = logging.getLogger(__name__)

# avoid to create temporary files to manage dataset (if read only)
from datasets import disable_caching

disable_caching()

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path_or_url: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        n_split: int = 1,
        split_id: int = 0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.n_split = n_split
        if n_split > 1 and split_id >= n_split:
            raise ValueError(
                f"split_id must be smaller than n_split. Got {split_id} and {n_split}"
            )
        self.split_id = split_id
        if isinstance(dataset_path_or_url, str):
            # check if it's a local path
            if Path(dataset_path_or_url).is_dir():
                self.dataset = load_from_disk(dataset_path_or_url)
            else:
                self.dataset = load_dataset(dataset_path_or_url)
        else:
            raise ValueError(
                f"dataset_path_or_url must be a string, got {type(dataset_path_or_url)}"
            )
        # the dataset from huggingface might be a DatasetDict with only one split
        if isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset["train"]

    def split_dataset(self):
        logger.info(
            f"Splitting the dataset in {self.n_split} parts, taking part {self.split_id}"
        )
        logger.info(f"Original dataset size: {len(self.dataset)}")
        if self.n_split > 1:
            idx = np.arange(len(self.dataset))
            splits = [[] for _ in range(self.n_split)]
            # if we launch with one split and the inference crashes,
            # we can easily relaunch on multi gpu without problem where one process will be useless
            # because all is already generated
            for id in idx:
                splits[id % self.n_split].append(id)
            self.dataset = self.dataset.select(splits[self.split_id])
        logger.info(f"Dataset size after split: {len(self.dataset)}")

    def filter_already_processed_prompts(self, processed_prompts):
        logger.info(
            f"Filtering out {len(processed_prompts)} already processed prompts."
        )
        self.dataset = self.dataset.filter(
            lambda x: x["prompt"] not in processed_prompts
        )

    def setup(self, stage: str = "test") -> None:
        self.dataset_test = self.dataset

        # convert bboxes columns if exists to string
        def convert_bboxes(x):
            if "bboxes" in x:
                new_bboxes = {}
                for k, bbox in x["bboxes"].items():
                    if bbox is None:
                        new_bboxes[k] = []
                    else:
                        new_bboxes[k] = [str(v) for v in bbox]
                x["bboxes_str"] = new_bboxes
            return x

        if "bboxes" in self.dataset_test.column_names:
            self.dataset_test = self.dataset_test.map(convert_bboxes)
            # drop the bboxes column
            self.dataset_test = self.dataset_test.remove_columns("bboxes")

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_prompt_list(self):
        return list(self.dataset["prompt"])
