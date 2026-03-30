from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore


class ClipScoreEval:
    def __init__(
        self,
        url_or_path_clipscore: Optional[str] = None,
        url_or_path_qualityscore: Optional[str] = None,
    ) -> None:
        self.device = torch.device("cpu")
        if url_or_path_clipscore is not None:
            self.clipscore_model = CLIPScore(url_or_path_clipscore, compute_on_cpu=True)
            self.clipscore_model.to(self.device)
        else:
            self.clipscore_model = None

        if url_or_path_qualityscore is not None:
            self.qualityscore_model = CLIPImageQualityAssessment(url_or_path_qualityscore, compute_on_cpu=True)
            self.prompt_qualityscore = ("quality",)
            self.qualityscore_model.to(self.device)
        else:
            self.qualityscore_model = None

    def compute(self):
        if self.clipscore_model is not None:
            clipscore = self.clipscore_model.compute()
        else:
            clipscore = None

        if self.qualityscore_model is not None:
            qualityscore = self.qualityscore_model.compute()
        else:
            qualityscore = None
        return clipscore, qualityscore

    def reset(self):
        if self.clipscore_model is not None:
            self.clipscore_model.reset()
        if self.qualityscore_model is not None:
            self.qualityscore_model.reset()

    @torch.no_grad()
    def __call__(
        self,
        images: Union[torch.Tensor, np.ndarray],
        prompts: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        elif isinstance(images, torch.Tensor):
            pass
        else:
            raise ValueError("img must be a numpy array or a torch tensor")
        images = images.to(self.device)

        if self.clipscore_model is not None:
            if prompts is None:
                raise ValueError("prompts is required")
            self.clipscore_model.update(images, prompts)
        if self.qualityscore_model is not None:
            self.qualityscore_model.update(images)

    def to(self, device):
        self.device = device
        if self.clipscore_model is not None:
            self.clipscore_model.to(device)
        if self.qualityscore_model is not None:
            self.qualityscore_model.to(device)


class ClipScoreEvalPerPrompt(ClipScoreEval):
    def __init__(
        self,
        save_dir: Optional[str] = None,
        url_or_path_clipscore: Optional[str] = None,
        url_or_path_qualityscore: Optional[str] = None,
    ) -> None:
        super().__init__(url_or_path_clipscore, url_or_path_qualityscore)
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    def compute_for_prompt(self, prompt: str):
        clipscore, qualityscore = self.compute()
        data = {}
        if clipscore is not None:
            data["clipscore"] = clipscore.detach().cpu().numpy()
        if qualityscore is not None:
            if len(self.prompt_qualityscore) > 1:
                for prompt_clip in qualityscore.keys():
                    data[prompt_clip] = qualityscore[prompt_clip].mean().detach().cpu().numpy()
                    data[prompt_clip + "_per_image"] = qualityscore[prompt_clip].detach().cpu().numpy().tolist()
            else:
                prompt_clip = self.prompt_qualityscore[0]
                data[prompt_clip] = qualityscore.mean().detach().cpu().numpy()
                data[prompt_clip + "_per_image"] = qualityscore.detach().cpu().numpy().tolist()
        data["prompt"] = prompt
        self.reset()
        if self.save_dir is not None:
            # create a dataframe
            pd.DataFrame(data).to_json(self.save_dir / f"{'_'.join(prompt.split())}.json", indent=4)
            # save results to file
        return data

    def get_data_from_multiple_files(self, save_dir: Optional[str] = None):
        """Load the data from the json files and save the results in a json file

        Args:
            save_dir (Optional[str], optional): directory to save the results. Defaults to None.
        """
        # get all the json files
        files = list(self.save_dir.glob("*.json"))
        # load and concat all the json files
        all_scores = []
        for f in files:
            all_scores.append(pd.read_json(f))
        all_scores = pd.concat(all_scores, ignore_index=True)
        if save_dir is not None:
            save_dir = Path(save_dir)
            if save_dir.is_dir():
                save_dir = save_dir / "clipscore.json"
            if not save_dir.suffix == ".json":
                save_dir = save_dir.with_suffix(".json")
            all_scores.to_json(save_dir, indent=4)

        columns_to_select = [col for col in all_scores.columns if not col.endswith("_per_image") and col != "prompt"]

        return all_scores[columns_to_select].mean(axis=0).to_frame().T

    def average_score(self, dataframe):
        columns_to_select = [col for col in dataframe.columns if not col.endswith("_per_image") and col != "prompt"]

        return dataframe[columns_to_select].mean(axis=0).to_frame()

    def score_per_seed(self, dataframe):
        pass
