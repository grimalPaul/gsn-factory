import torch
from t2v_metrics.constants import CONTEXT_LEN
from t2v_metrics.models.vqascore_models.clip_t5_model import CLIP_T5_MODELS, CLIPT5Model


class VQAScoreCustom(CLIPT5Model):
    def __init__(self, path_tokenizer, path_model):
        self.model_name = "clip-flant5-xxl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_loader = lambda image: image
        self.cache_dir = None
        CLIP_T5_MODELS["clip-flant5-xxl"] = {
            "tokenizer": {
                "path": path_tokenizer,
                "model_max_length": CONTEXT_LEN,
            },
            "model": {
                "path": path_model,
                "conversation": "t5_chat",
                "image_aspect_ratio": "pad",
            },
        }
        self.load_model()

    def __call__(self, images, texts):
        return self.forward(images, texts)
