# reimplementation of https://github.com/yuval-alaluf/Attend-and-Excite/tree/main/metrics

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from ..utils import RunningMean, RunningMeanDict
from .utils import COCO_TYPE, imagenet_templates


class AEEval:
    def __init__(
        self,
    ):
        self.reset()

    def reset(self):
        self.scores = {}
        self.scores = {
            "full_text_similarities_per_seed": RunningMeanDict(),
            "average_full_text_similarities": RunningMean(),
            "min_part_text_similarities_per_seed": RunningMeanDict(),
            "average_min_part_text": RunningMean(),
            "blip_caption_per_seed": {},  # to drop when we concatenate results
            "text_similarities_per_seed": RunningMeanDict(),
            "text_similarities": RunningMean(),
            "combination_labels": None,
        }

    def set_values_from_df(self, df: DataFrame):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    def save_score_all_seeds(
        self,
        seeds,
        full_text_similarities,
        min_part_similarities,
        text_similarities,
        blip_captions,
        combination_label,
    ):
        for i, seed in enumerate(seeds):
            self.save_score(
                seed,
                full_text_similarities[i],
                min_part_similarities[i],
                text_similarities[i],
                blip_captions[i],
                combination_label,
            )

    def save_score(
        self,
        seed,
        full_text_similarity,
        min_part_similarity,
        text_similarity,
        blip_caption,
        combination_labels,
    ):
        # per seed
        self.scores["full_text_similarities_per_seed"].update(full_text_similarity, seed)

        self.scores["min_part_text_similarities_per_seed"].update(min_part_similarity, seed)
        self.scores["text_similarities_per_seed"].update(text_similarity, seed)
        self.scores["blip_caption_per_seed"][seed] = blip_caption

        self.scores["average_full_text_similarities"].update(full_text_similarity)
        self.scores["average_min_part_text"].update(min_part_similarity)
        self.scores["text_similarities"].update(text_similarity)
        # average
        self.scores["combination_labels"] = combination_labels

    def mean_per_seed(self, key, seed, df):
        return df[key].apply(lambda x: x[seed]).mean()

    def var_on_all_data(self, key, df):
        scores = []
        for scores_seed in df[key].values:
            for _, score in scores_seed.items():
                scores.append(score)
        return np.var(scores)

    def process_df(self, df: DataFrame, c: Optional[str] = None):
        if c is not None:
            df_ = df[df["combination_labels"] == c]

        else:
            df_ = df
        seeds = list(df_["full_text_similarities_per_seed"].iloc[0].keys())
        to_drop = [
            "blip_caption_per_seed",
            "combination_labels",
            "full_text_similarities_per_seed",
            "min_part_text_similarities_per_seed",
            "text_similarities_per_seed",
        ]
        if "prompt" in df_.columns:
            to_drop.append("prompt")

        # mean per seed
        mean_full_text_similarities_per_seed = [
            self.mean_per_seed("full_text_similarities_per_seed", s, df_) for s in seeds
        ]
        mean_min_part_text_similarities_per_seed = [
            self.mean_per_seed("min_part_text_similarities_per_seed", s, df_) for s in seeds
        ]
        mean_text_similarities_per_seed = [self.mean_per_seed("text_similarities_per_seed", s, df_) for s in seeds]

        full_text_similiraties_var = self.var_on_all_data("full_text_similarities_per_seed", df_)
        min_part_text_similarities_var = self.var_on_all_data("min_part_text_similarities_per_seed", df_)
        text_similarities_var = self.var_on_all_data("text_similarities_per_seed", df_)

        df_ = df_.drop(columns=to_drop)

        df_ = df_.mean().to_frame().T
        df_["full_text_similarities_per_seed"] = [mean_full_text_similarities_per_seed]
        df_["min_part_text_similarities_per_seed"] = [mean_min_part_text_similarities_per_seed]
        df_["text_similarities_per_seed"] = [mean_text_similarities_per_seed]
        df_["combination_labels"] = [c]

        df_["full_text_similiraties_var"] = [full_text_similiraties_var]
        df_["min_part_text_similarities_var"] = [min_part_text_similarities_var]
        df_["text_similarities_var"] = [text_similarities_var]

        return df_

    def save_df(self, df: DataFrame, name: str, save_dir: str):
        if save_dir is not None:
            save_dir_score = Path(save_dir)
            if save_dir_score.is_dir():
                save_dir_score = save_dir_score / f"{name}.json"
                df.to_json(save_dir_score, indent=4)
            elif save_dir_score.is_file():
                # remove file part
                save_dir_score = save_dir_score.parent / f"{name}.json"
                df.to_json(save_dir_score, indent=4)
            else:
                print("save_dir is not a file or a directory")

    def compile_data_from_multiple_files(self, path_to_json_files, save_dir: Optional[str] = None):
        """Load the data from the json files and save the results in a json file

        Args:
            path_to_json_files (str): path to the json files
            save_dir (Optional[str], optional): directory to save the results. Defaults to None.
        """
        path_to_json_files = Path(path_to_json_files)
        if not path_to_json_files.is_dir():
            raise ValueError("path_to_json_files must be a directory")
        # get all the json files
        files = list(path_to_json_files.glob("*.json"))
        # load and concat all the json files
        all_scores = []

        for f in files:
            try:
                prompt = " ".join(f.stem.split("_"))
                df = pd.read_json(f)
                df["prompt"] = prompt
                all_scores.append(df)
            except Exception as e:
                print(f"Error with {f}: {e}")
        # Create a general df compiling all the information without dropping anything
        # Then return a df per combination of labels and general mean
        all_scores = pd.concat(all_scores, ignore_index=True)
        # save the concatenation
        if save_dir is not None:
            save_dir_score = Path(save_dir)
            if save_dir_score.is_dir():
                save_dir_score = save_dir_score / "image_text_sim.json"
            if not save_dir_score.suffix == ".json":
                save_dir_score = save_dir_score.with_suffix(".json")
            all_scores.to_json(save_dir_score, indent=4)
        # then compute and return some stats
        combination_labels = all_scores["combination_labels"].unique()
        df_per_combination = []
        for c in combination_labels:
            n_prompt = all_scores[all_scores["combination_labels"] == c].shape[0]
            df_per_combination.append(self.process_df(all_scores, c=c))
            df_per_combination[-1]["n_prompt"] = n_prompt
        df_per_combination = pd.concat(df_per_combination, ignore_index=True, axis=0)
        self.save_df(df_per_combination, "image_text_sim_combinations", save_dir)
        average_score = self.process_df(all_scores)
        average_score["n_prompt"] = all_scores.shape[0]
        self.save_df(average_score, "image_text_sim_average", save_dir)
        return average_score, df_per_combination

    def compile_score_from_dataframe(self, dataframe: DataFrame):
        combination_labels = dataframe["combination_labels"].unique()
        df_per_combination = []
        for c in combination_labels:
            n_prompt = dataframe[dataframe["combination_labels"] == c].shape[0]
            df_per_combination.append(self.process_df(dataframe, c=c))
            df_per_combination[-1]["n_prompt"] = n_prompt
        df_per_combination = pd.concat(df_per_combination, ignore_index=True, axis=0)
        average_score = self.process_df(dataframe)
        average_score["n_prompt"] = dataframe.shape[0]
        return average_score, df_per_combination


class AEEvalPerPrompt(AEEval):
    def __init__(
        self,
        save_dir: str = None,
        url_or_path_clip: str = "openai/clip-vit-base-patch16",
        url_or_path_blip: str = "Salesforce/blip-image-captioning-base",
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device("cpu")

        # clip
        self.clip_model = CLIPModel.from_pretrained(url_or_path_clip)
        self.clip_processor = CLIPProcessor.from_pretrained(url_or_path_clip)
        self.clip_model.to(self.device)

        # blip
        self.blip_processor = BlipProcessor.from_pretrained(url_or_path_blip)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(url_or_path_blip)
        self.blip_model.to(self.device)

        self.batch_size = batch_size
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    def stream_batch_size(self, batch: torch.Tensor, seeds: List[int]):
        """return a generator that yield batch of images
        Args:
            batch (torch.Tensor): NCHW tensor
        """
        n = batch.size(0)
        for i in range(0, n, self.batch_size):
            yield batch[i : min(i + self.batch_size, n)], seeds[i : min(i + self.batch_size, n)]

    def get_type(self, classe: str):
        return COCO_TYPE[classe]

    @torch.no_grad()
    def __call__(
        self,
        images,
        classes: List[str],
        prompt: str,
        seeds_used: Optional[List[int]] = None,
        colors: Optional[List[str]] = None,
    ):
        """
        Receive all the images associated with the prompt and the classes
        """
        self.reset()
        if not prompt.startswith("a photo of"):
            print(
                "Warning: prompt not start with 'a photo of', we consider that is a template like in Attend and Excite"
            )
            prompt_truncated = prompt
        else:
            prompt_truncated = prompt.replace("a photo of ", "")

        type_per_label = "_".join([self.get_type(c) for c in classes])

        full_text_features = self.get_embedding_for_prompt(prompt=prompt_truncated, templates=imagenet_templates)
        if colors is not None:
            part_features = torch.stack(
                [
                    self.get_embedding_for_prompt(
                        prompt=f"{colors[classe]} {classe}",
                        templates=imagenet_templates,
                    )
                    for classe in classes
                ],
                dim=0,
            )
        else:
            part_features = torch.stack(
                [self.get_embedding_for_prompt(prompt=classe, templates=imagenet_templates) for classe in classes],
                dim=0,
            )

        images_features = self.get_embedding_images(images)

        # compute the score
        # full text similarities
        full_text_similarities = torch.matmul(images_features, full_text_features).cpu().numpy().tolist()

        # part text similarities
        part_txt_similarities = torch.matmul(images_features, part_features.mT)
        # min part text similarities
        min_part_txt_similarities = part_txt_similarities.min(dim=1).values.cpu().numpy().tolist()

        # blip captionning
        blip_captions = self.generate_caption(images)
        blip_embeddings = self.clip_embedding(blip_captions)
        text_similarities = torch.matmul(blip_embeddings, full_text_features).cpu().numpy().tolist()

        # save data
        # save generated captions
        self.save_score_all_seeds(
            combination_label=type_per_label,
            blip_captions=blip_captions,
            seeds=seeds_used,
            full_text_similarities=full_text_similarities,
            min_part_similarities=min_part_txt_similarities,
            text_similarities=text_similarities,
        )
        if self.save_dir is not None:
            self.save_prompt_scores_to_json(prompt)

        return self.create_dict_results(prompt)

    def get_embedding_images(self, images):
        inputs = self.clip_processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    def clip_embedding(self, texts):
        inputs = self.clip_processor(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        embeddings = self.clip_model.get_text_features(**inputs)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def get_embedding_for_prompt(self, prompt, templates):
        texts = [template.format(prompt) for template in templates]  # format with class
        texts = [t.replace("  ", " ") for t in texts]  # remove double space
        texts = [t.replace(" a a ", " a ") for t in texts]  # remove double a's
        texts = [t.replace("the a", "a") for t in texts]  # remove double a's
        class_embeddings = self.clip_embedding(texts)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        return class_embedding.float()

    def generate_caption(self, image):
        inputs = self.blip_processor(images=image, return_tensors="pt", padding=True).to(self.device)
        caption = self.blip_model.generate(**inputs)
        caption = self.blip_processor.batch_decode(caption, skip_special_tokens=True)
        return caption

    def create_dict_results(self, prompt):
        columns = {
            "full_text_similarities_per_seed": [self.scores["full_text_similarities_per_seed"].get()],
            "average_full_text_similarities": [self.scores["average_full_text_similarities"].get()],
            "min_part_text_similarities_per_seed": [self.scores["min_part_text_similarities_per_seed"].get()],
            "average_min_part_text": [self.scores["average_min_part_text"].get()],
            "blip_caption_per_seed": [self.scores["blip_caption_per_seed"]],
            "text_similarities_per_seed": [self.scores["text_similarities_per_seed"].get()],
            "text_similarities": [self.scores["text_similarities"].get()],
            "combination_labels": [self.scores["combination_labels"]],
            "prompt": prompt,
        }
        return columns

    def save_prompt_scores_to_json(self, prompt):
        file_name = self.save_dir / f"{'_'.join(prompt.split())}.json"
        columns = self.create_dict_results(prompt)
        DataFrame(columns).to_json(file_name, indent=4)

    def to(self, device):
        self.device = device
        self.clip_model.to(device)
        self.blip_model.to(device)
