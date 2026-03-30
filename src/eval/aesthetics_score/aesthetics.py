from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from aesthetics_predictor import AestheticsPredictorV2Linear
from transformers import CLIPProcessor

# shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE


class AestheticsScore:
    def __init__(self, url_or_path_aesthetics: str = None) -> None:
        self.device = torch.device("cpu")
        if url_or_path_aesthetics is not None:
            self.aesthetics_model = AestheticsPredictorV2Linear.from_pretrained(url_or_path_aesthetics)
            self.processor = CLIPProcessor.from_pretrained(url_or_path_aesthetics)
            self.aesthetics_model.to(self.device)
        else:
            self.aesthetics_model = None
        self.score = []

    def to(self, device):
        self.device = device
        self.aesthetics_model.to(device)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        if self.aesthetics_model is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.aesthetics_model(**inputs)
            prediction = outputs.logits
            self.score.extend(prediction.detach().cpu().numpy().flatten())

    def compute(self):
        if len(self.score) > 0:
            mean = sum(self.score) / len(self.score)
            return mean, self.score
        else:
            return None, None

    def reset(self):
        self.score = []


class AestheticsScorePerPrompt(AestheticsScore):
    def __init__(self, save_dir: str = None, url_or_path_aesthetics: str = None) -> None:
        super().__init__(url_or_path_aesthetics)
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    def compute_for_prompt(self, prompt: str):
        data = {}
        mean, per_image = self.compute()
        if mean is not None:
            data["aesthetics_score"] = mean
            data["aesthetics_score_per_image"] = per_image
            data["prompt"] = prompt
            # create a dataframe
            self.reset()
            if self.save_dir is not None:
                pd.DataFrame(data).to_json(self.save_dir / f"{'_'.join(prompt.split())}.json", indent=4)
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
        if len(all_scores) != 0:
            all_scores = pd.concat(all_scores, ignore_index=True)
        else:
            return None
        if save_dir is not None:
            save_dir = Path(save_dir)
            if save_dir.is_dir():
                save_dir = save_dir / "aesthetics_score.json"
            if not save_dir.suffix == ".json":
                save_dir = save_dir.with_suffix(".json")
            all_scores.to_json(save_dir, indent=4)

        columns_to_select = [col for col in all_scores.columns if not col.endswith("_per_image") and col != "prompt"]

        return all_scores[columns_to_select].mean(axis=0).to_frame().T

    def average_score(self, dataframe):
        columns_to_select = [col for col in dataframe.columns if not col.endswith("_per_image") and col != "prompt"]
        return dataframe[columns_to_select].mean(axis=0).to_frame().T

    def score_for_seed(self, dataframe, seed):
        pass


"""
créer des json pour chaque prompt

Puis tout concaténer à la fin

"""
