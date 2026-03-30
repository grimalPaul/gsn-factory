from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
from PIL import Image

from ..eval import (
    AEEvalPerPrompt,
    AestheticsScorePerPrompt,
    ClipScoreEvalPerPrompt,
    VQAScoreCustom,
)
from ..utils import RankedLogger
from .utils import ConcurrentWriter, get_object_list

logger = RankedLogger(__name__, rank_zero_only=True)


class ScoreModule(L.LightningModule):
    def __init__(
        self,
        save_dir: str,  # where to save score per prompt and load images
        path_or_url_clipscore: Optional[str] = None,
        path_or_url_qualityscore: Optional[str] = None,
        path_or_url_aesthetics: Optional[str] = None,
        path_or_url_image_text_sim_blip: Optional[str] = None,
        path_or_url_image_text_sim_clip: Optional[str] = None,
        params_vqa_model: Optional[Dict] = None,
        n_images: int = 16,
        seeds: Optional[List[int]] = None,
        prompts_to_generate: Optional[List[str]] = None,
        **kwargs,
    ):
        L.LightningModule.__init__(self)
        if seeds is not None:
            if not isinstance(seeds, list) or not all(
                isinstance(seed, int) for seed in seeds
            ):
                raise ValueError("Seeds must be a list of integers")
            if n_images != len(seeds):
                raise ValueError(
                    "Number of seeds must be equal to number of desired images to generate"
                )
            self.seeds = seeds
        else:
            self.seeds = list(range(n_images))
        self.n_images = n_images
        self.save_dir = Path(save_dir)
        # check if save_dir exists
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory {self.save_dir}")
        else:
            logger.info(
                f"Directory {self.save_dir} already exists. In case of conflict, files will be overwritten."
            )

        # Folder index
        self.index_writer = ConcurrentWriter(
            self.save_dir / "index.txt", type_data="text"
        )
        self.prompt_already_processed = self.index_writer.read_all_lines()

        if prompts_to_generate is not None:
            self.all_is_processed = set(prompts_to_generate) == set(
                self.prompt_already_processed
            )
        else:
            self.all_is_processed = False
        print(f"self.all_is_processed: {self.all_is_processed}")
        # images folder
        self.images_writer = ConcurrentWriter(
            path=self.save_dir / "images",
            type_data="image",
        )

        # Metrics
        ## CLIP
        if path_or_url_clipscore is not None or path_or_url_qualityscore is not None:
            self.clipscore_writer = ConcurrentWriter(
                self.save_dir / "clipscore.tar", type_data="json"
            )
            self.clipscore = ClipScoreEvalPerPrompt(
                url_or_path_clipscore=path_or_url_clipscore,
                url_or_path_qualityscore=path_or_url_qualityscore,
            )
        else:
            self.clipscore = None

        ## Aesthetic
        if path_or_url_aesthetics is not None:
            self.aesthetics_writer = ConcurrentWriter(
                path=self.save_dir / "aesthetics.tar", type_data="json"
            )
            self.aesthetics = AestheticsScorePerPrompt(
                url_or_path_aesthetics=path_or_url_aesthetics,
            )
        else:
            self.aesthetics = None

        ## AE evaluation
        if (
            path_or_url_image_text_sim_blip is not None
            and path_or_url_image_text_sim_clip is not None
        ):
            self.image_text_sim_writer = ConcurrentWriter(
                self.save_dir / "image_text_sim.tar", type_data="json"
            )
            self.image_text_sim = AEEvalPerPrompt(
                url_or_path_blip=path_or_url_image_text_sim_blip,
                url_or_path_clip=path_or_url_image_text_sim_clip,
            )
        else:
            self.image_text_sim = None
        ## VQA
        logger.info(f"VQA{params_vqa_model}")
        if params_vqa_model is not None:
            save_tar_vqa = self.save_dir / "vqa_score.tar"
            self.vqa_writer = ConcurrentWriter(save_tar_vqa, type_data="json")
            self.vqa_score_model = VQAScoreCustom(
                path_tokenizer=params_vqa_model["tokenizer"],
                path_model=params_vqa_model["model"],
            )
            logger.info(self.vqa_score_model)
            # Enable to pass directly PIL image
            self.vqa_score_model.image_loader = lambda image: image
        else:
            self.vqa_score_model = None
        self.multi_template_style_prompt = False

    def on_test_start(self) -> None:
        self.load_pipeline_on_gpu()

    def load_pipeline_on_gpu(self):
        if self.clipscore is not None:
            self.clipscore.to(self.device)
        if self.image_text_sim is not None:
            self.image_text_sim.to(self.device)
        if self.aesthetics is not None:
            self.aesthetics.to(self.device)
        # if self.vqa_score_model is not None:
        #     self.vqa_score_model.to(self.device)

    def test_step(self, batch, batch_idx):
        batch_size = len(batch["prompt"])
        prompt = batch["prompt"]
        list_objects, color_objects = get_object_list(batch)
        print(prompt)
        if color_objects == []:
            color_classes = [None] * batch_size
        else:
            adjs = list(batch["adj_apply_on"].keys())
            color_classes = []
            for i in range(batch_size):
                colors_from_prompt = {}
                for adj in adjs:
                    word = batch["adj_apply_on"][adj][i]
                    colors_from_prompt[batch["labels_params"][word][i]] = batch[
                        "adjs_params"
                    ][adj][i]
                color_classes.append(colors_from_prompt)

        for i, prompt_ in enumerate(prompt):
            if prompt_ in self.prompt_already_processed:
                if not self.multi_template_style_prompt:
                    self.multi_template_style_prompt = any(
                        obj == "" for obj in list_objects
                    )
                list_objects = [obj for obj in list_objects if obj != ""]
                # images are available
                images = self.get_images_from_tar(
                    names=[f"{'_'.join(prompt_.split())}_{s}.png" for s in self.seeds]
                )
                images = self.pil_to_torch(images)
                self.compute_scores(
                    images=images,
                    list_objects=list_objects[i],
                    color_classes=color_classes[i],
                    prompt=prompt_,
                )

    def pil_to_torch(self, images):
        return torch.stack(
            [
                torch.tensor(np.array(im)).permute(2, 0, 1).to(torch.float32) / 255.0
                for im in images
            ]
        )

    def on_test_epoch_end(self):
        self.save_scores()

    def torch_to_pil(self, images):
        return [
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            for img in images
        ]

    def _compute_vqa_score(self, prompt: List[str], images: List[Image.Image]):
        images = self.torch_to_pil(images)
        scores = self.vqa_score_model(images=images, texts=[prompt] * len(images))
        logger.info(scores)
        data = {}
        data["vqascore"] = scores.mean().detach().cpu().numpy()
        data["vqascore_per_image"] = scores.detach().cpu().numpy().tolist()
        data["prompt"] = prompt
        self.vqa_writer.write_json(data=data, name=self.get_file_name(prompt))

    def compute_scores(self, prompt, images, list_objects, color_classes):
        if self.clipscore is not None:
            self._compute_clipscore(prompt, images)

        if self.aesthetics is not None:
            self._compute_aesthetics_score(prompt, images)

        if self.image_text_sim is not None:
            self._compute_image_text_sim(prompt, images, list_objects)

        if self.vqa_score_model is not None:
            self._compute_vqa_score(prompt, images)

    def get_file_name(self, prompt):
        return f"{'_'.join(prompt.split())}.json"

    def _compute_clipscore(self, prompt, images):
        images_for_clip = self.images_float_to_uint8(images)
        self.clipscore(images_for_clip, [prompt] * images_for_clip.shape[0])
        score_dict = self.clipscore.compute_for_prompt(prompt)
        self.clipscore_writer.write_json(
            data=score_dict, name=self.get_file_name(prompt)
        )

    def _compute_aesthetics_score(self, prompt, images):
        images_for_aesthetics = self.images_float_to_uint8(images)
        self.aesthetics(images_for_aesthetics)
        score_dict = self.aesthetics.compute_for_prompt(prompt)
        self.aesthetics_writer.write_json(
            data=score_dict, name=self.get_file_name(prompt)
        )

    def _compute_image_text_sim(self, prompt, images, list_objects):
        images_for_image_text_sim = self.images_float_to_uint8(images)
        score_dict = self.image_text_sim(
            images=images_for_image_text_sim,
            classes=list_objects,
            prompt=prompt,
            seeds_used=self.seeds,
        )
        self.image_text_sim_writer.write_json(
            data=score_dict, name=self.get_file_name(prompt)
        )

    def save_scores(self):
        if self.clipscore is not None:
            dataframe = self.clipscore_writer.read_all_json()
            dataframe = dataframe.reset_index(drop=True)
            dataframe.to_json(self.save_dir / "clipscore.json", indent=4)
            average_score = self.clipscore.average_score(dataframe)
            logger.info(f"\n{average_score}")
            average_score.to_markdown(f"{self.save_dir}/clipscore.md", index=True)

        if self.aesthetics is not None:
            dataframe = self.aesthetics_writer.read_all_json()
            dataframe = dataframe.reset_index(drop=True)
            dataframe.to_json(self.save_dir / "aesthetics.json", indent=4)
            average_score = self.aesthetics.average_score(dataframe)
            logger.info(f"\n{average_score}")
            average_score.to_markdown(f"{self.save_dir}/aesthetics.md", index=True)

        if self.image_text_sim is not None:
            dataframe = self.image_text_sim_writer.read_all_json()
            dataframe = dataframe.reset_index(drop=True)
            (
                average_score,
                df_per_combination,
            ) = self.image_text_sim.compile_score_from_dataframe(dataframe)
            df_per_combination.to_json(
                self.save_dir / "image_text_sim_combinations.json", indent=4
            )
            average_score.to_json(
                self.save_dir / "image_text_sim_average.json", indent=4
            )
        if self.vqa_score_model is not None:
            dataframe = self.vqa_writer.read_all_json()
            dataframe = dataframe.reset_index(drop=True)
            dataframe.to_json(self.save_dir / "vqa_score.json", indent=4)
            columns_to_select = [
                col
                for col in dataframe.columns
                if not col.endswith("_per_image") and col != "prompt"
            ]
            agg_vqa_score = dataframe[columns_to_select].mean(axis=0).to_frame().T
            agg_vqa_score.to_markdown(f"{self.save_dir}/vqa_score.md", index=True)
            logger.info(f"\n{agg_vqa_score}")

    def images_float_to_uint8(self, images):
        return torch.clip(images * 255, 0, 255).to(torch.uint8)

    def get_images_from_tar(self, names):
        return self.images_writer.get_images_(names)
