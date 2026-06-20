# FiNDR: Vocabulary-Free Fine-Grained Recognition via Reasoning-Augmented LMMs (CVPR 2026)

This repository contains the reference implementation for **FiNDR (Fine-grained Name Discovery via Reasoning)**, a fully automated framework for **vocabulary-free fine-grained image recognition** using reasoning-augmented Large Multi-Modal Models (LMMs).

> **Thinking Beyond Labels: Vocabulary-Free Fine-Grained Recognition using Reasoning-Augmented LMMs**
> Dmitry Demidov, Zaigham Zaheer, Zongyan Han, Omkar Thawakar, Rao Anwer
> Mohamed bin Zayed University of Artificial Intelligence
> [[arXiv]](https://arxiv.org/abs/2512.18897)
> [[CVPR 2026 (soon)]]()

FiNDR removes the need for predefined or human-curated label vocabularies by discovering, verifying, and using fine-grained semantic class names directly from unlabelled images. Our approach challenges the assumption that human-defined vocabularies represent an upper bound for fine-grained recognition performance by outperforming zero-shot classifiers that rely on ground-truth class names.

---

## Overview

Traditional fine-grained recognition methods rely on fixed, human-defined label vocabularies, which limits scalability and robustness in open-world scenarios. FiNDR addresses this limitation by leveraging modern **reasoning-capable large multi-modal models** together with **vision–language models** to automatically induce fine-grained class names directly from visual data.

The entire pipeline operates **without any predefined vocabulary, manual annotation, or supervised training**.

---

## ✨ Key Contributions

- Introduces the **first reasoning-augmented LMM framework** for vocabulary-free fine-grained image recognition.
- Proposes a **fully automated end-to-end pipeline** that discovers, verifies, and uses semantic class names.
- Achieves **state-of-the-art performance** across multiple fine-grained benchmarks in the vocabulary-free setting.
- **Outperforms zero-shot classifiers** that rely on ground-truth class names, challenging the assumption that human-curated vocabularies represent an upper bound.
- Demonstrates that **open-source LMMs**, with carefully designed prompts, can match proprietary reasoning-enabled models.

---

## Method Summary

Given a small unlabelled discovery set, FiNDR operates in three stages:

### 1. Vocabulary Discovery via Reasoning
A reasoning-enabled LMM:
- Infers dataset-level meta-information (e.g., category, granularity, domain expertise)
- Generates fine-grained candidate class names for each image using step-by-step reasoning

### 2. Class Name Refinement
A vision–language model (e.g., CLIP):
- Measures visual–semantic alignment between images and candidate names
- Filters and ranks candidate labels to form a refined vocabulary

### 3. Vision–Language Modalities Coupling
Textual and visual prototypes are combined into a lightweight vision–language classifier, which is used at inference time to assign **human-readable fine-grained labels** to unseen images.

---

## Usage

### 1. Setup Environment

#### 1.1a Set up environment using PIP:

1. Install Python 3.9.16 (skip if installed already):
```bash
# installing Python with conda, but you can use any other method
conda create -n findr python=3.9.16 -y
conda activate findr
```

2. Install dependencies with PIP:
```bash
pip install -r envs/pip_requirements.txt
```

#### 1.1b Alternatively, set up environment using Conda:

1. Install dependencies with Conda:
```bash
conda env create -f envs/conda_environment.yml
# or
conda create --name findr --file envs/conda_requirements.txt
```

2. Activate the environment and install CLIP (non-conda package):
```bash
conda activate findr
pip install git+https://github.com/openai/CLIP.git
```

#### 1.2 Install dependencies for a chosen LMM:

a. For **Qwen-VL** (used in our experiments):
```bash
pip install openai
```

b. For **Gemini**:
```bash
pip install google-genai
```

c. For **ChatGPT**:
```bash
pip install openai
```

d. For other LMMs, please refer to their respective repositories for installation instructions.

### 2. Prepare Datasets

The datasets are available here: [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/dmitry_demidov_mbzuai_ac_ae/IgDfCVJKmoRzTbFb_z5eX10MAR1CIDKQzt69QGA37Zgq0Jk)

Additionally, you can use direct download links:
- **Birds-200** (CUB-200-2011):
```bash
wget --content-disposition "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/IQDk4hHO2i_-S7q_A6eG-h9MAWl7HdxXlow6UsRdhQcEyhI?download=1"
```

- **Cars-196** (Stanford Cars):
```bash
wget --content-disposition "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/IQAolhbGZpdZRpr2DG8J5nZQAWfgHWB-S2pASn1K-cvjKHY?download=1"
```

- **Dogs-120** (Stanford Dogs):
```bash
wget --content-disposition "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/IQDARv5F02pfQZsTlSrcPSaeAcdCWIXPExurhFJxNMJroNk?download=1"
```

- **Flowers-102** (Oxford Flowers):
```bash
wget --content-disposition "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/IQD4r2FdG3nfRacgXuAIrIImAReuFRdrr8LyZd_yYIiT0U4?download=1"
```

- **Pets-37** (Oxford Pets):
```bash
wget --content-disposition "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/dmitry_demidov_mbzuai_ac_ae/IQD8wqlmsNf7QYURZ1B9Ko1SAYJTP3BssiOl5aKndjhoKk4?download=1"
```

Alternatively, for dataset download and preparation, follow a beautifully written guide available [here](https://github.com/OatmealLiu/FineR?tab=readme-ov-file#-datasets-preparation).

All meta data needed for the supported datasets is provided in the `data/data_stats.py` file.

### 3. Classname Discovery
The discovered class names are already provided in the `data/guessed_classnames/` directory for all supported datasets.

\[Optionally\] To re-discover clasnames, run: 
```bash
# For a custom dataset - update generation config in the script
python -m data.generate_classnames
```

### 4. Vocabulary-free Classification
To perform vocabulary-free classification on supported datasets, run the corresponding evaluation scripts:

```bash
sh run/eval_birds.sh
sh run/eval_cars.sh
sh run/eval_dogs.sh
sh run/eval_flowers.sh
sh run/eval_pets.sh
```

---

## Repository Structure

```
findr/
├── configs/               # Configuration files for experiments
├── data/                  # Dataset loaders, preprocessing, generated in-context sentences
├── datasets/              # Fine-grained datasets
├── envs/                  # Environment setup files
├── models/                # Vision-language interfaces
├── run/                   # Entry-point scripts for experiments
├── utils/                 # Helper utilities
└── README.md
```

---

## Citation

If you find this work useful, consider a citation:

```
@misc{demidov2025thinkinglabelsvocabularyfreefinegrained,
      title={Thinking Beyond Labels: Vocabulary-Free Fine-Grained Recognition using Reasoning-Augmented LMMs}, 
      author={Dmitry Demidov and Zaigham Zaheer and Zongyan Han and Omkar Thawakar and Rao Anwer},
      year={2025},
      eprint={2512.18897},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.18897}, 
}
```

---

## Contacts

For questions or collaborations:

* **Dmitry Demidov** – [dmitry.demidov@mbzuai.ac.ae](mailto:dmitry.demidov@mbzuai.ac.ae)

---

## ⭐ Acknowledgements

This project builds upon and integrates ideas from [CLIP](https://github.com/openai/CLIP), [FineR](https://github.com/OatmealLiu/FineR), [E-FineR](https://github.com/demidovd98/e-finer), and recent advances in reasoning-based LMMs ([Qwen](https://github.com/QwenLM/Qwen2.5-VL), [Gemini](https://arxiv.org/abs/2507.06261)).
We are thankful to the corresponding authors for making their code public.
