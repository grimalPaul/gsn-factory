from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from ..utils import RankedLogger
from .gsn_config import DistribConfig, GsngConfig, IterefConfig
from .gsn_criterion import BoxDiffGSN, RetentionLoss, SynGen
from .pipeline import StableDiffusion3PipelineGSN, StableDiffusionGSN
from .utils import ConcurrentWriter, get_object_list

AVAIL_PIPELINE = {
    "sd3": StableDiffusion3PipelineGSN,
    "sd14": StableDiffusionGSN,
}

logger = RankedLogger(__name__, rank_zero_only=True)


def pt_to_numpy(images: torch.FloatTensor):
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


class ImageGenerationPipeline(L.LightningModule):
    def __init__(
        self,
        pipeline_url_or_path: str,
        save_dir: str,  # where to save score per prompt and images
        name: str,
        params_pipeline_inference: Optional[Dict[str, Any]] = None,
        enable_vae_slicing: bool = True,
        torch_dtype: Optional[str] = "32",
        n_images: int = 16,
        seeds: Optional[List[int]] = None,
        batch_size_pipeline: Optional[int] = None,
        scheduler=None,
        iteref_config: Optional[IterefConfig] = None,
        gsng_config: Optional[GsngConfig] = None,
        distrib_config: Optional[DistribConfig] = None,
        drop_eot: bool = False,
        clip_start_index_for_syngen: Optional[int] = None,
        t5_start_index_for_syngen: Optional[int] = None,
        distrib_seed: List[int] = [0],  # used only with the distrib approach
        prompts_to_generate: Optional[List[str]] = None,
    ):
        L.LightningModule.__init__(self)

        super().__init__()
        if seeds is not None:
            if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
                raise ValueError("Seeds must be a list of integers")
            if n_images != len(seeds):
                raise ValueError("Number of seeds must be equal to number of desired images to generate")
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
            logger.info(f"Directory {self.save_dir} already exists. In case of conflict, files will be overwritten.")

        # Folder index
        self.index_writer = ConcurrentWriter(self.save_dir / "index.txt", type_data="text")
        self.prompt_already_processed = self.index_writer.read_all_lines()

        # images folder
        self.images_writer = ConcurrentWriter(
            path=self.save_dir / "images",
            type_data="image",
        )
        self.save_hyperparameters(logger=False)

        if isinstance(prompts_to_generate, list):
            self.all_is_processed = set(prompts_to_generate) == set(self.prompt_already_processed)
            if self.all_is_processed:
                logger.info("All the images are already generated")
        else:
            self.all_is_processed = False

        if not self.all_is_processed:
            self.distrib_seed = distrib_seed

            if batch_size_pipeline is not None:
                self.batch_size_pipeline = batch_size_pipeline
            else:
                logger.warning(f"batch_size_pipeline not set, set to {n_images=}")
                self.batch_size_pipeline = n_images

            if name not in AVAIL_PIPELINE:
                raise ValueError(f"Pipeline {name} not found in {AVAIL_PIPELINE.keys()}")
            self.name = name
            if torch_dtype == "32":
                dtype = torch.float32
            elif torch_dtype == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            logger.info(f"run model with dtype {dtype}")
            self.pipeline = AVAIL_PIPELINE[name].from_pretrained(pipeline_url_or_path, torch_dtype=dtype)
            self.pipeline.safety_checker = None

            if scheduler is not None:
                self.pipeline.scheduler = scheduler.from_config(self.pipeline.scheduler.config)
                logger.info(f"Scheduler set to {self.pipeline.scheduler}")

            if params_pipeline_inference is not None:
                self.pipeline_inference_params = params_pipeline_inference
            else:
                self.pipeline_inference_params = OmegaConf.create({})

            if self.is_sd14_or_sd3():
                self.iteref_config = iteref_config
                self.gsng_config = gsng_config
                self.distrib_config = distrib_config
                self.drop_eot = drop_eot
                self.clip_start_index_for_syngen = clip_start_index_for_syngen
                self.t5_start_index_for_syngen = t5_start_index_for_syngen
                self.pipeline_inference_params["iteref_config"] = self.iteref_config
                self.pipeline_inference_params["gsng_config"] = self.gsng_config
                self.pipeline_inference_params["distrib_config"] = self.distrib_config
            else:
                self.iteref_config = None
                self.gsng_config = None
                self.distrib_config = None
                self.drop_eot = None

            # cast self.pipeline_inference_params to correct type
            self.pipeline_inference_params = OmegaConf.to_container(self.pipeline_inference_params)

            if enable_vae_slicing and not self.is_sd3():
                self.pipeline.enable_vae_slicing()

    def is_sd3(self):
        return self.name == "sd3"

    def is_sd14_or_sd3(self):
        return self.name in ["sd14", "sd3"]

    def on_test_start(self) -> None:
        self.load_pipeline_on_gpu()

    def load_pipeline_on_gpu(self):
        self.pipeline.to(self.device)
        if self.is_sd3():
            self.pipeline.enable_model_cpu_offload()

    def get_objects_bbox(self, params):
        def transform_coordinates(input_dict):
            transformed = {}

            for obj_name, coordinates in input_dict.items():
                # Check if the coordinates are empty
                if coordinates == []:
                    continue

                # Determine number of coordinates per point
                num_coords = len(coordinates[0])

                # Initialize empty lists for each coordinate
                coord_lists = [[] for _ in range(num_coords)]

                # Distribute values to appropriate lists
                for point in coordinates:
                    for i in range(num_coords):
                        coord_lists[i].append(float(point[i]))

                transformed[obj_name] = coord_lists

            return transformed

        logger.info(f"params {params['bboxes_str']}")
        list_bboxes = transform_coordinates(params["bboxes_str"])
        logger.info(f"list_bboxes {list_bboxes}")
        bboxes = [[] for _ in range(len(list_bboxes["object1"]))]
        for bbox in list_bboxes.values():
            for j, v in enumerate(bbox):
                bboxes[j].append(v)

        logger.info(f"bboxes {bboxes}")
        return bboxes

    def convert_list_tensor_to_float(self, list_tensor):
        result = []
        for data in list_tensor:
            if isinstance(data, torch.Tensor):
                result.append(data.detach().cpu().numpy().tolist())
            else:
                result.append(self.convert_list_tensor_to_float(data))
        return result

    def get_extras_params(
        self,
        prompt,
        list_objects,
        color_objects,
        bboxes=None,
        extra_params={},
    ):
        """
        Prepare the extra parameters for the pipeline inference.

        """

        if self.is_sd14_or_sd3():
            if self.iteref_config is not None:
                extra_params["iteref_config"] = self.iteref_config
                extra_params["iteref_config"].extra_params = self.get_params_criterion(
                    prompt=prompt,
                    entity_list=list_objects,
                    criterion=self.iteref_config.criterion,
                    adj_list=color_objects,
                )
                if bboxes is not None:
                    if isinstance(self.iteref_config.criterion, RetentionLoss):
                        self.iteref_config.criterion.masks_available = True
                        logger.info(f"all the bboxes {bboxes}")
                        extra_params["iteref_config"].extra_params["bboxes"] = bboxes
                    if isinstance(self.iteref_config.criterion, BoxDiffGSN):
                        extra_params["iteref_config"].extra_params["bboxes"] = bboxes

            if self.distrib_config is not None:
                extra_params["distrib_config"] = self.distrib_config
                extra_params["distrib_config"].extra_params = self.get_params_criterion(
                    prompt=prompt,
                    entity_list=list_objects,
                    criterion=self.distrib_config.criterion,
                    adj_list=color_objects,
                )
                if bboxes is not None:
                    if isinstance(self.distrib_config.criterion, RetentionLoss):
                        self.distrib_config.criterion.masks_available = True
                        logger.info(f"all the bboxes {bboxes}")
                        extra_params["distrib_config"].extra_params["bboxes"] = bboxes
                    if isinstance(self.distrib_config.criterion, BoxDiffGSN):
                        extra_params["distrib_config"].extra_params["bboxes"] = bboxes

            if self.gsng_config is not None:
                extra_params["gsng_config"] = self.gsng_config
                extra_params["gsng_config"].extra_params = self.get_params_criterion(
                    prompt=prompt,
                    entity_list=list_objects,
                    criterion=self.gsng_config.criterion,
                    adj_list=color_objects,
                )
                if bboxes is not None:
                    if isinstance(self.gsng_config.criterion, RetentionLoss):
                        self.gsng_config.criterion.masks_available = True
                        logger.info(f"all the bboxes {bboxes}")
                        extra_params["gsng_config"].extra_params["bboxes"] = bboxes
                    if isinstance(self.gsng_config.criterion, BoxDiffGSN):
                        extra_params["gsng_config"].extra_params["bboxes"] = bboxes

        return extra_params

    def remove_indices(self, data, indices_to_remove):
        if isinstance(data, list):
            return [v for i, v in enumerate(data) if i not in indices_to_remove]
        elif isinstance(data, dict):
            return {k: self.remove_indices(v, indices_to_remove) for k, v in data.items()}
        else:
            return data

    def get_params_criterion(
        self,
        prompt,
        entity_list,
        criterion,
        adj_list,
    ):
        # prepare arguments according to the type of criterion
        extra_params = {
            "start_idx_clip": [],
            "last_idx_clip": [],
            "token_indices_clip": [],
        }

        if self.is_sd3():
            extra_params.update(
                {
                    "start_idx_t5": [],
                    "last_idx_t5": [],
                    "token_indices_t5": [],
                }
            )

        if isinstance(criterion, SynGen):
            extra_params["adjs_token_clip"] = []
            if self.is_sd3():
                extra_params["adjs_token_t5"] = []
        for prompt_, entity_list_, adj_list_ in zip(prompt, entity_list, adj_list):
            token_indices = self.pipeline.get_attention_idx_from_token(prompt_, entity_list_)
            if adj_list_ is not None:
                adj_indices = self.pipeline.get_attention_idx_from_token(prompt_, adj_list_)
            else:
                adj_indices = None
            if self.drop_eot or isinstance(criterion, SynGen):
                last_idx = self.pipeline.get_indice_last_token(prompt_)
            else:
                last_idx = [-1, -1] if self.is_sd3() else -1
            if self.is_sd3():
                token_indices_clip = token_indices[0]
                token_indices_t5 = token_indices[1]
                if adj_indices is not None:
                    adj_indices_clip = adj_indices[0]
                    adj_indices_t5 = adj_indices[1]
                last_idx_clip = last_idx[0]
                last_idx_t5 = last_idx[1]

                extra_params["last_idx_t5"].append(last_idx_t5)
            else:
                token_indices_clip = token_indices
                adj_indices_clip = adj_indices
                last_idx_clip = last_idx
                token_indices_t5 = None
                adj_indices_t5 = None
                last_idx_t5 = None
            extra_params["last_idx_clip"].append(last_idx_clip)

            if isinstance(criterion, SynGen) and adj_indices_clip is not None:
                if len(adj_indices_clip) != len(token_indices_clip):
                    raise ValueError(
                        "The number of entities and the number of group of attributes must be the same. Outside this pipeline you can align the entity without attribute with an empty list. It is not possible here."
                    )
                adj_indices_clip = [[i] for i in adj_indices_clip]
                if adj_indices_t5 is not None:
                    adj_indices_t5 = [[i] for i in adj_indices_t5]
                if self.is_sd3():
                    extra_params["start_idx_t5"].append(self.t5_start_index_for_syngen)
                    extra_params["adjs_token_t5"].append(adj_indices_t5)
                extra_params["adjs_token_clip"].append(adj_indices_clip)
                extra_params["start_idx_clip"].append(self.clip_start_index_for_syngen)
            else:
                extra_params["start_idx_clip"].append(1)
                if self.is_sd3():
                    extra_params["start_idx_t5"].append(1)
            if self.is_sd3():
                extra_params["token_indices_t5"].append(token_indices_t5)
            extra_params["token_indices_clip"].append(token_indices_clip)
        if "adjs_token_clip" in extra_params:
            if len(extra_params["adjs_token_clip"]) == 0:
                # delete key if empty
                del extra_params["adjs_token_clip"]
                if self.is_sd3():
                    del extra_params["adjs_token_t5"]
        return extra_params

    def test_step(self, batch, batch_idx):
        batch_size = len(batch["prompt"])
        index_2_remove = []
        for i, p in enumerate(batch["prompt"]):
            if p in self.prompt_already_processed:
                index_2_remove.append(i)
        if len(index_2_remove) == batch_size:
            return None  # skip the batch if all the prompts have already been processed

        if index_2_remove:
            # Recursively remove the unwanted indices from all elements in the batch
            for key in list(batch.keys()):
                batch[key] = self.remove_indices(batch[key], index_2_remove)

        prompt = batch["prompt"]
        batch_size = len(prompt)
        if "bboxes_str" in batch:
            bboxes = self.get_objects_bbox(batch)
        else:
            bboxes = None
        logger.info(f"batch: {batch}, remove {len(index_2_remove)} prompts")
        list_objects, color_objects = get_object_list(batch)

        if color_objects == []:
            color_objects = [None] * batch_size
            color_classes = [None] * batch_size
        else:
            adjs = list(batch["adj_apply_on"].keys())
            color_classes = []
            for i in range(batch_size):
                colors_from_prompt = {}
                for adj in adjs:
                    word = batch["adj_apply_on"][adj][i]
                    colors_from_prompt[batch["labels_params"][word][i]] = batch["adjs_params"][adj][i]
                color_classes.append(colors_from_prompt)

        extra_params = self.get_extras_params(
            extra_params=self.pipeline_inference_params.copy(),
            prompt=prompt,
            list_objects=list_objects,
            color_objects=color_objects,
            bboxes=bboxes,
        )

        logger.info(f"extra_params: {extra_params}")
        batch_generators = []
        seeds = self.seeds * batch_size
        batch_seeds = self.create_packing_list(seeds, self.batch_size_pipeline)

        for b_ in batch_seeds:
            batch_generators.append([torch.Generator(self.device).manual_seed(s) for s in b_])

        if batch_size > 1 and self.batch_size_pipeline < batch_size * self.n_images:
            raise ValueError(
                "If batch size is greater than 1, the batch_size_pipeline must be >= batch size x n_images. The batch size cannot be splited."
            )

        images_stack = []
        for generators in batch_generators:
            if self.distrib_config is not None:
                extra_params["generator_distrib"] = [
                    torch.Generator(device=self.device).manual_seed(s)
                    for s in self.distrib_seed
                    for _ in range(batch_size)  # one seed per distribution to be learned
                ]

            images_stack.append(
                self.pipeline(
                    prompt=prompt,
                    num_images_per_prompt=(self.n_images if len(generators) >= self.n_images else len(generators)),
                    generator=generators,
                    output_type="pt",  # return NCHW tensor with "pt"
                    **extra_params,
                )
                .images.detach()
                .cpu()
            )
        images = torch.cat(images_stack, dim=0)

        for i, p in enumerate(prompt):
            self.save_images(
                names=[f"{'_'.join(p.split())}_{s}.png" for s in self.seeds],
                images=self.torch_to_Image(images[i * self.n_images : (i + 1) * self.n_images]),
            )
            self.index_writer.write_text(p)

    def torch_to_Image(self, tensor):
        return numpy_to_pil(pt_to_numpy(tensor))

    def create_packing_list(self, objects: List, n: int):
        """
        Create a list of list of size n from a list of objects
        """
        return [objects[i : i + n] for i in range(0, len(objects), n)]

    def save_images(self, names, images):
        self.images_writer.write_images(images=[(name, img) for name, img in zip(names, images)])
