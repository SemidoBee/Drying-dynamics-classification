# Measurement-Informed Deep Learning for Drying Dynamics Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
*(https://doi.org/10.5281/zenodo.20694590)*

This repository contains the official dataset and source code for the deep learning models, computational efficiency evaluation, and Explainable AI (XAI) analysis presented in our paper, recently accepted for publication in ***ACS Applied Materials & Interfaces***.

## 1. Data Availability (Zenodo)
Due to GitHub's file size limitations for large datasets, the image datasets required to run this code are hosted on Zenodo. 

Please download the following two zip files from our Zenodo repository: **[https://doi.org/10.5281/zenodo.20694590]**
*   `Original_Image.zip`: Contains the random-split dataset.
*   `Holdout_Image.zip`: Contains the strictly isolated independent drying runs dataset used during our revision process to robustly validate model generalizability.

After downloading, extract them into the root directory of this repository to match the structure below.

## 2. Repository Structure
Ensure your directory is organized as follows before running the codes:

```text
├── Original_Image/                       # Directory containing the original image dataset (Random Split)
│   ├── train/                            # Training images (5 classes: A, B, C, alpha, beta)
│   ├── validation/                       # Validation images (5 classes: A, B, C, alpha, beta)
│   └── test/                             # Test images (5 classes: A, B, C, alpha, beta)
├── Holdout_Image/                        # Directory containing the independent hold-out dataset (Strictly Isolated)
│   ├── train/                            # Training images (5 classes: A, B, C, alpha, beta)
│   ├── validation/                       # Validation images (5 classes: A, B, C, alpha, beta)
│   └── test/                             # Test images (5 classes: A, B, C, alpha, beta)
├── Drying_Classifier_CNN.py              # Code for model training and evaluation
├── Classifier_Parameters_Analysis.py     # Code for evaluating FLOPs, Parameters, and Inference Speed
├── Classifier_XAI_Analysis.py            # Code for extracting Grad-CAM and Input Gradient Saliency maps
├── requirements.txt                      # List of required Python packages
└── README.md                             # This instruction file
```

---

## 3. Environment Setup
To ensure full reproducibility and avoid version conflicts, we highly recommend using an Anaconda virtual environment.

**Step 1:** Open your Anaconda Prompt (or terminal).  
**Step 2:** Create and activate a new virtual environment (Python 3.8+ recommended):
```bash
conda create -n drying_ai_env python=3.9 -y
conda activate drying_ai_env
```
**Step 3:** Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Usage

### A. Model Training and Evaluation (`Drying_Classifier_CNN.py`)
To train and evaluate the model, you can specify the target dataset (`Original_Image` or `Holdout_Image`) using the command-line arguments. 

**Example execution for the Original dataset:**
```bash
python Drying_Classifier_CNN.py --train_dir "./Original_Image/train" --val_dir "./Original_Image/validation" --test_dir "./Original_Image/test" --output_dir "./results_original"
```

**Example execution for the independent Hold-out dataset:**
```bash
python Drying_Classifier_CNN.py --train_dir "./Holdout_Image/train" --val_dir "./Holdout_Image/validation" --test_dir "./Holdout_Image/test" --output_dir "./results_holdout"
```

The trained model weights (`best_model.pth`) and performance metrics (Confusion Matrix, Accuracy/Loss curves) will be saved in the specified output directory.

### B. Computational Efficiency Analysis (`Classifier_Parameters_Analysis.py`)
To validate the theoretical efficiency (16.4x reduction in computational cost) and measure the real-time inference speed (Latency/FPS) of the proposed CNN architecture, run:

```bash
python Classifier_Parameters_Analysis.py
```
*(This script runs a standalone validation using dummy tensors to measure FLOPs and latency on your specific hardware.)*

### C. Explainable AI (XAI) Analysis (`Classifier_XAI_Analysis.py`)
To generate the visual explanations (Input Gradient Saliency map and Grad-CAM overlay) for a specific test image, run the following command. Note that you must provide the path to a trained model weight (e.g., from step A) and the target image.

**Example execution:**
```bash
python Classifier_XAI_Analysis.py --image_path "./Original_Image/test/A/new_image_24 (5).jpg" --model_weight "./results_original/best_model.pth" --output_dir "./xai_results"
```
*(Tip: If your file path contains spaces, please make sure to enclose the path in double quotation marks `" "` as shown above.)*

The generated heatmaps will be saved in the `./xai_results` directory.

---
## Citation
If you find this dataset or code useful for your research, please consider citing our paper:
*(Citation details will be updated once the paper is published online.)*

**Contact:** For any technical issues or inquiries regarding the dataset/code, please contact the corresponding author.
