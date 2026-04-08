# Physics-Augmented Deep Learning for Drying Dynamics Classification

This repository contains the dataset and source code for reproducing the deep learning models, computational efficiency evaluation, and Explainable AI (XAI) analysis presented in our Nature Communications submission.

## 1. Repository Structure
Please ensure the downloaded files are organized in the following structure before running the codes:

```
├── Image_data/                           # Directory containing the image dataset
│   ├── train/                            # Training images
│   ├── validation/                       # Validation images
│   └── test/                             # Test images
├── Drying_Classifier_CNN.py              # Code for model training and evaluation
├── CLassifier_Parameters_Analysis.py     # Code for evaluating FLOPs, Parameters, and Inference Speed
├── Classifier_XAI_Analysis.py            # Code for extracting Grad-CAM and Input Gradient Saliency maps
├── requirements.txt                      # List of required Python packages
└── README.md                             # This instruction file
```
---

## 2. Environment Setup
To ensure full reproducibility and avoid version conflicts, we highly recommend using an Anaconda virtual environment.

**Step 1:** Open your Anaconda Prompt (or terminal).
**Step 2:** Create and activate a new virtual environment (Python 3.8+ recommended):
```bash
conda create -n drying_ai_env python=3.9 -y
conda activate drying_ai_env
```
**Step 3:** Install all required dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Note: A CUDA-enabled GPU is highly recommended for faster training and inference, though the code automatically falls back to CPU if no GPU is detected.)*

---

## 3. Running the Codes

All scripts are designed to be executed via the command line interface using `argparse` for maximum flexibility. Please run the commands from the root directory of this repository.

### A. Model Training (`Drying_Classifier_CNN.py`)
To train the lightweight CNN model from scratch using the provided image dataset, run the following command. The trained model weights (`best_model.pth`) and confusion matrices will be saved in the `./results` directory.

```bash
python Drying_Classifier_CNN.py --train_dir "./Image_data/train" --val_dir "./Image_data/validation" --test_dir "./Image_data/test" --output_dir "./results"
```

### B. Computational Efficiency Analysis (`CLassifier_Parameters_Analysis.py`)
To validate the theoretical efficiency (16.4x reduction in computational cost) and measure the real-time inference speed (Latency/FPS) of the proposed CNN architecture, run:

```bash
python CLassifier_Parameters_Analysis.py
```
*(This script runs a standalone validation using dummy tensors to measure FLOPs and latency on your specific hardware.)*

### C. Explainable AI (XAI) Analysis (`Classifier_XAI_Analysis.py`)
To generate the visual explanations (Input Gradient Saliency map and Grad-CAM overlay) for a specific test image, run the following command. Note that you must provide the path to a trained model weight (e.g., from step A) and the target image.

**Example execution:**
```bash
python Classifier_XAI_Analysis.py --image_path "./Image_data/test/A/new_image_24 (5).jpg" --model_weight "./results/best_model.pth" --output_dir "./xai_results"
```
*(Tip: If your file path contains spaces, please make sure to enclose the path in double quotation marks `" "` as shown above.)*

The generated heatmaps will be saved in the `./xai_results` directory.

---
**Contact:** For any technical issues or inquiries regarding the dataset/code, please contact the corresponding author.
