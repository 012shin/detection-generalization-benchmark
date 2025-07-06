# Detection Generalization Benchmark

A comprehensive benchmark framework for evaluating generalization performance of various backbone-detector combinations in object detection.

## 🎯 Overview

**Detection Generalization Benchmark** is a research framework designed to systematically evaluate the generalization capabilities of different backbone and detector combinations in object detection tasks. This project is based on the hypothesis that **"heterogeneous combinations between backbones and detectors may cause generalization performance degradation"**.

## 🔬 Research Background

Object detection models consist of two main components: **backbone** and **detector**, which have different learning objectives, data distributions, and processing methods:

### Detector Characteristics by Type:
- **1-stage detectors** (e.g., YOLO, RetinaNet)
- **2-stage detectors** (e.g., Faster R-CNN, Cascade R-CNN)
- **End-to-end detectors** (e.g., DETR, DiffDet)

### Backbone Characteristics:
- **CNN-based** (ResNet, ConvNeXt, InternImage)
- **Transformer-based** (ViT, Swin, CLIP)

This research experimentally validates the hypothesis that when these differences are combined, especially in combinations where characteristics conflict, generalization performance may unexpectedly decrease.

## ✨ Key Features

### 1. Diverse Model Combinations
- **CNN Backbones**: ResNet, ConvNeXt, InternImage
- **Transformer Backbones**: ViT, Swin, CLIP
- **1-stage Detectors**: FocusNet, RetinaNet
- **2-stage Detectors**: Faster R-CNN, Cascade R-CNN
- **End-to-end Detectors**: DETR, DiffDet

### 2. Robustness Evaluation
Evaluates performance not only on clean COCO data but also on realistic environmental variations:
- Fog, low lighting conditions
- Noise, resolution degradation
- Other realistic image corruptions

### 3. Flexible Training Strategies
- **Full fine-tuning (ft)**: Train entire model
- **Parameter-efficient tuning (peft)**: Adapter-based training
- **Layer-wise tuning**: Train specific layers only
- **Decoder probing**: Train decoder only

## 🛠️ Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1
- CUDA-compatible GPU (recommended)
- OpenCV (optional, for demo and visualization)

### Setup Instructions

git clone https://github.com/012shin/detection-generalization-benchmark.git
cd detection-generalization-benchmark



2. **Install Detectron2**
For Linux (pre-compiled version)
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

Or build from source
cd detectron2
python -m pip install -e .

text

3. **Install project dependencies**
cd projects
python setup.py install


## 🚀 Usage

### 1. Training

Basic training command:
cd projects
python tools/train_net.py --config-file configs/[detector]/[backbone]/config.yaml --num-gpus 8


### 2. Evaluation

Evaluate trained models:
python tools/train_net.py --config-file configs/[detector]/[backbone]/config.yaml --eval-only MODEL.WEIGHTS path/to/model.pth

text

### 3. Robustness Testing

Evaluate robustness against image corruptions:
python tools/corruption_test.py --config-file configs/[detector]/[backbone]/config.yaml MODEL.WEIGHTS path/to/model.pkl



## 📄 License

This project is distributed under the Apache 2.0 License. See `LICENSE` file for details.

## 📚 Citation

If you use this benchmark in your research, please cite:

@misc{detection-generalization-benchmark,
title={Detection Generalization Benchmark},
author={Korea University},
year={2024},
url={https://github.com/012shin/detection-generalization-benchmark}
}


## 📞 Contact

For questions or bug reports about this project, please submit through GitHub Issues.

---

## 🔍 Evaluation Metrics

- **mAP (mean Average Precision)**: Standard object detection metric
- **Robustness Score**: Performance under various corruptions
- **Generalization Gap**: Difference between clean and corrupted performance
