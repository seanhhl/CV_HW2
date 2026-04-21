# NYCU Computer Vision 2026 HW2

- Student ID：112261014
- Name：李瀚翔

## Introduction
This repository contains the PyTorch implementation for NYCU Computer Vision 2026 HW2: Digit Detection. The goal is to detect digits in RGB images and predict their bounding boxes and classes.
To effectively improve the model's performance and handle small object detection, we applied several advanced techniques:

* Multi-Scale Training
* Robust Data Augmentation (via albumentations)
* Focal Loss & Increased GIoU Weight
* Automatic Mixed Precision (AMP) for efficient training

---

## Environment Setup
This project is implemented and fully tested on **Google Colab**.

### Basic Requirements:
* **Python:** 3.8+
* **Libraries:** `torch`, `torchvision`, `transformers`, `albumentations`, `Pillow`, `tqdm`
* **Hardware:** CUDA-enabled GPU (e.g., Colab T4/L4 GPU)

---

## Usage

To train the DETR model from scratch and generate the predictions, simply run the main script (or execute the cells sequentially in Google Colab):
```bash
python train.py
```

## Snapshot
<img width="1157" height="40" alt="image" src="https://github.com/user-attachments/assets/0d779c30-1b6f-4fd0-ba33-585052df9ee0" />

