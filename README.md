# SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation (AAAI 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2508.13866-b31b1b.svg)](https://arxiv.org/abs/2508.13866)
[![Conference](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/)

This repository contains the official implementation of the paper **"SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation"**, accepted at **AAAI 2026**.

> State-of-the-art text-to-image models produce visually impressive results but often struggle with precise alignment to text prompts, leading to missing critical elements or unintended blending of distinct concepts. We propose a novel approach that learns a high-success-rate distribution conditioned on a target prompt, ensuring that generated images faithfully reflect the corresponding prompts. Our method explicitly models the signal component during the denoising process, offering fine-grained control that mitigates over-optimization and out-of-distribution artifacts. Moreover, our framework is training-free and seamlessly integrates with both existing diffusion and flow matching architectures. It also supports additional conditioning modalities — such as bounding boxes — for enhanced spatial alignment. Extensive experiments demonstrate that our approach outperforms current state-of-the-art methods.

We build upon the GSN (Generative Semantic Nursing) approach introduced by Chefer et al. (arXiv:2301.13826) and extend it with our SAGA method.

The GSN field is rapidly evolving, with numerous methods leveraging attention maps during inference to refine image generation. We designed this codebase to facilitate the comparison of existing methods and the development of custom GSN losses. We hope this framework assists the community in implementing these techniques across new architectures.

Note: Contributions are welcome! Parts of the code are still being refined for efficiency.

## Quick start

### Environment setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the environment.

Create a new environment and install the dependencies with:

```bash
# Create a virtual environment with Python 3.13+
uv venv --python 3.13
# install the dependencies
uv sync
```

This will install all the necessary packages to run the experiments and notebooks.

## Running Experiments

### Data preparation

If internet access is available during runtime, datasets will download automatically by passing `Paulgrimal/DATASET_NAME` in your generation command.

For environments without internet access:

1. Download the required datasets manually.
2. Store them in a local directory.
3. Update the `dataset_dir` field in `configs/paths/default.yaml` to point to your local directory path.

```python
from datasets import load_dataset

to_dl = [
    "2_entities_bbox",
    "3_entities_bbox",
    "2_entities",
    "3_entities",
    "4_entities",
]
dataset_dir = "datasets/" # change if needed

for name in to_dl:
    load_dataset(f"Paulgrimal/{name}").save_to_disk(f"{dataset_dir}/{name}")
```

### Model Setup

To run generation and evaluation, download the models listed below and place them in a local directory (e.g., `models/`). Update the `model_dir` path in `configs/paths/default.yaml` to point to this directory. Alternatively, you can point `model_dir` to your existing Hugging Face cache folder.

#### Generative Models

* **SD 1.4:** [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
* **SD 3:** [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)

#### Evaluation Models

* **CLIP:** [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16), **save the model in a path containing `PATH_TO_MODELS/openai/clip-vit-base-patch16`** because the evaluation use `torchmetrics` and need to have `openai`in the path.
* **BLIP** (for the image text similarity score of attend and Excite): [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
* **Aesthetics Predictor:** [shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE](https://huggingface.co/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE)

* **VQA Model Configuration** For VQA, download the following three components:
  1. [zhiqiulin/clip-flant5-xxl](https://huggingface.co/zhiqiulin/clip-flant5-xxl)
  2. [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)
  3. [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
  * **Required Change:** In the `config.json` of **clip-flant5-xxl**, update the `mm_vision_tower` field to the local path of the downloaded CLIP model:
`"mm_vision_tower": "YourPath/clip-vit-large-patch14-336"`

### Image Generation

Use the `demo.ipynb` notebook to test generation and visualize results interactively.

Execute experiments using specific configuration files located in `configs/experiment/aaai/`.

**Local Execution:**

```bash
python src/generate.py experiment=aaai/sd14/saga_one_distrib
```

**Cluster Execution:**

For cluster environments, this project supports the [Submitit Hydra Launcher](https://hydra.cc/docs/plugins/submitit_launcher/).

1. **Configure Launcher**: update configs/hydra/launcher/submitit_example.yaml with your environment activation commands and cluster settings.

2. **Select Cluster Config:** Use experiment files designed for clusters, such as:
   * `configs/experiment/aaai/sd14/example_on_cluster_saga_one_distrib.yaml`
   * `configs/experiment/aaai/sd3/example_on_cluster_saga_one_distrib.yaml`

3. **Launch Job**: After activating your environment, run the following command to queue the job:

```bash
python src/generate.py experiment=aaai/sd14/example_on_cluster_saga_one_distrib
```

### Evaluation

You can run the evaluation script by providing the dataset path and the output directory of your generated images. The script expects images in `save_dir/images` and an `index.txt` file in `save_dir`.

**Direct Command:**

```bash
python src/evaluate.py dataset_url_or_path=PATH_TO_DATASET save_dir=PATH_TO_IMAGES

```

**Using Config Files:**

```bash
python src/evaluate.py experiment=aaai/evaluation

```

#### Cluster Evaluation (Submitit)

To run evaluation on a cluster, use a dedicated configuration file (e.g., `configs/experiment/aaai/evaluation_example_on_cluster.yaml`):

```bash
python src/evaluate.py experiment=aaai/evaluation_example_on_cluster

```

#### Requirements & Constraints

* **Dataset Integrity:** Evaluation cannot be split. All generated images for the dataset must be in `save_dir/images` and the `index.txt` file must be fully processed.
* **Config Parameters:** Ensure your `data` parameters are set as follows:

```yaml
data:
    n_split: 1
    batch_size: 1
    split_id: 0

```

#### TIAM and GenEval Evaluation

For other metrics, use the following repositories:

* **TIAM Score:** [CEA-LIST/TIAMv2](https://github.com/CEA-LIST/TIAMv2)
* **GenEval:** [djghosh13/geneval](https://github.com/djghosh13/geneval)
* Use these Hugging Face datasets for GenEval generation:
* `Paulgrimal/GenEval_prompts_wo_color`
* `Paulgrimal/GenEval_prompts_w_one_color`
* `Paulgrimal/GenEval_prompts_w_two_colors`

* After generation, run the scoring code directly from the GenEval repository.
  
### Baselines

The following implementations are used for baseline comparisons. For consistent and fair evaluation, ensure all baselines use the **DDPM scheduler**.

* **Attend-and-Excite:** [yuval-alaluf/Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite)
* **InitNO:** [xiefan-guo/initno](https://github.com/xiefan-guo/initno)
* **SynGen:** [RoyiRa/Linguistic-Binding-in-Diffusion-Models](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models)

## More details about the codebase

### Concepts

In this part we explain the different concepts implemented and how they are articulated.

#### GSN Guidance (GSNGg)

*Chefer et al.* [arxiv:2301.13826](https://arxiv.org/abs/2301.13826) introduced GSN guidance. This involves shifting the latent image $x_t$ once per diffusion step during the first half of the sampling process to maintain image quality. The shift is applied using a gradient descent step on the latent image $x_t$ : $x_{t} \leftarrow x_t - \alpha_{t} \cdot \nabla_{x_t} \mathcal{L}$, with $\alpha_{t}$, the learning rate.

#### Iterative Refinement (Iteref)

The GSNg process can be repeated at each of some predefined sampling steps $t_1\dots t_k$ until either a loss threshold is met or a maximum number of shifts is reached. Thresholds must be carefully calibrated for each step.

#### SAGA, learn a distribution aligned with the signal

We present two variants of our SAGA method. In the first variant, a single distribution is learned, and multiple latent images are sampled from it. In the second variant, multiple distributions are learned, and one latent image is sampled from each distribution.

![one](https://arxiv.org/html/2508.13866v2/x13.png)

*One distribution is learned and N=3 latent images are sampled from it.*

![two](https://arxiv.org/html/2508.13866v2/x14.png)

*N = 3 distributions are learned and one latent image is sampled from each.*

### Custom GSN Loss Implementation

You can develop and test custom GSN losses by referencing the existing implementations in `src/gsn_criterion/`.

#### Steps to Implement

1. **Inherit from AbstractGSN:** Your new class must inherit from the `AbstractGSN` base class located in `src/gsn_criterion/utils`.
2. **Reference Template:** Use `src/models/gsn_criterion/attend_and_excite.py` as a minimal working example.
3. **Define the Loss:** Update the `function_loss` attribute to point to your custom loss function.
4. **Adjust Thresholds:** Modify the threshold logic as needed to define the specific values required to stop the optimization process.

### Criterion Details

* **Attend and Excite** (`attend_and_excite.py`): Cross-attention loss designed to align generated images with specific text prompts.
* **InitNO** (`initno.py`): An optimization approach based on latent initialization.
* **SynGen** (`syngen.py`): Implements linguistic binding through semantic decomposition.
  * **Token Filtering**: Improved results are achieved by excluding contextual tokens (e.g., "a photo of") that do not contribute to semantic alignment.
  * **Configuration**: Adjust `start_idx_clip` and `start_idx_t5` to set the starting token index.
    * **Default (1)**: Skips the "Start-Of-Text" token.
    * **Example (4)**: Use this for prompts like "a photo of [object]" to skip the first three contextual tokens.
* **IOU** (`iou.py`): Integrates attention-based excitation with Intersection-over-Union (IoU) constraints for precise spatial alignment.
* **Retention Loss** (`retention_loss.py`): An attention-based loss that supports optional spatial constraints via bounding boxes or attention-derived masks.
* **BoxDiff** (`boxdiff.py`): Provides spatial control using bounding boxes without requiring corner constraints.

> **Note:** To maintain consistency, all criteria in this repository have been reimplemented within our unified framework. However, baseline comparisons utilize the authors' original implementations to ensure a fair and accurate evaluation against published results.

## Parameter Configuration

For a hands-on guide on passing parameters to the loss function, refer to the `demo.ipynb` notebook. Below is a detailed breakdown of how to configure entity tracking and distribution settings.

### Entity Token Mapping

When generating images with specific objects (e.g., `"a photo of a cat and a dog"`), you must provide the token indices for the entities you wish to guide.

#### Single Prompt Example

```python
prompt = "a photo of a cat and a dog"
# Retrieve indices for "cat" and "dog"
token_indices, adj_indices, last_idx = get_token_indices(pipeline, prompt, ["cat", "dog"], None)
# Example output: token_indices = [[5], [8]] 

extra_params_gsng = {"token_indices_clip": [token_indices]}

gsng_config = GsngConfig(
    scale_factor=20,
    scale_range=[1, 1],
    steps=list(range(0, 25)),
    criterion=AttendAndExciteGSN(),
    extra_params=extra_params_gsng,
)

```

#### Multiple Prompts (Batched)

To process different prompts in a single batch, provide a list containing the token indices for each prompt:

```python
prompts = ["a photo of a cat and a dog", "a photo of a car and a tree"]
t_idx_1, _, _ = get_token_indices(pipeline, prompts[0], ["cat", "dog"], None)
t_idx_2, _, _ = get_token_indices(pipeline, prompts[1], ["car", "tree"], None)

extra_params_gsng = {"token_indices_clip": [t_idx_1, t_idx_2]}

```

### Seed and Generator Management

The number of **generator seeds** must match the total number of images you intend to generate.

* **1 Prompt, 4 Images:** Requires 4 generator seeds.
* **2 Prompts, 4 Images per prompt:** Requires 8 generator seeds.

### Distribution Configurations

The SAGA method relies on `DistribConfig` to manage how latent distributions are learned and sampled.

#### Single Distribution per Prompt (`one_image_per_distrib=False`)

In this setup, the model learns **one distribution** for a given prompt and samples multiple images from it.

* **Distribution Seeds:** Pass 1 seed per prompt.
* **Latent Seeds:** Pass seeds equal to the total number of images (e.g., for 2 prompts and 4 images each, pass 8 seeds).

```python
DistribConfig(
    ...
    one_image_per_distrib=False,
)

```

#### One Distribution per Image (`one_image_per_distrib=True`)

In this setup, the model learns a **unique distribution** for every individual image generated for a prompt.

* **Seeds:** The number of distribution seeds must exactly match the number of latent generator seeds.

```python
DistribConfig(
    ...
    one_image_per_distrib=True,
)

```

## Remarks

* **SD 2.0/2.1:** This version is theoretically supported but has not been extensively tested. You will likely need to adjust hyperparameters. Additionally, ensure you drop the End-Of-Text (EOT) token by providing the `last_idx` to the loss function.
* **Flux Model:** Preliminary implementation for Flux is available. However, further research is required to determine which specific blocks provide the most effective attention maps for guidance.
* **IOU Criterion:** This implementation is a hybrid of the Attend-and-Excite and IoU losses. While we have developed a batched function (`_compute_loss_batched`) to process attention maps specifically for SAGA, please note that **this implementation was not used for the experiments reported in the paper.**

## Contributing

Contributions are welcome! If you have suggestions for improvements or find areas where the code could be more efficient, feel free to open an issue or submit a pull request. We appreciate any feedback that helps refine the codebase and enhance project performance.

## Citation

If you find this code useful for your research, please consider citing our paper:

```
@misc{grimal2026sagalearningsignalaligneddistributions,
      title={SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation}, 
      author={Paul Grimal and Michaël Soumm and Hervé Le Borgne and Olivier Ferret and Akihiro Sugimoto},
      year={2026},
      eprint={2508.13866},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.13866}, 
}
```

## Acknowledgments

This codebase is built upon the following frameworks and libraries:

* **Lightning Hydra Template:** [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
* **Diffusers Library:** [huggingface/diffusers](https://github.com/huggingface/diffusers)
