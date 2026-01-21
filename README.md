# SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation

[![arXiv](https://img.shields.io/badge/arXiv-2508.13866-b31b1b.svg)](https://arxiv.org/abs/2508.13866)
[![Conference](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/)

This repository contains the official implementation of the paper **"SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation"**, accepted at **AAAI 2026**.

> **Status:** 🚧 The code for this project will be released soon. Please stay tuned!

> State-of-the-art text-to-image models produce visually impressive results but often struggle with precise alignment to text prompts, leading to missing critical elements or unintended blending of distinct concepts. We propose a novel approach that learns a high-success-rate distribution conditioned on a target prompt, ensuring that generated images faithfully reflect the corresponding prompts. Our method explicitly models the signal component during the denoising process, offering fine-grained control that mitigates over-optimization and out-of-distribution artifacts. Moreover, our framework is training-free and seamlessly integrates with both existing diffusion and flow matching architectures. It also supports additional conditioning modalities — such as bounding boxes — for enhanced spatial alignment. Extensive experiments demonstrate that our approach outperforms current state-of-the-art methods.
