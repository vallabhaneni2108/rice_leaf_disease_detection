# Rice Leaf Disease Detection

A deep learning project for classifying rice leaf images into three
disease categories using convolutional neural networks (CNNs) and
transfer learning with MobileNetV2.

## Project Overview

The project explores image preprocessing, exploratory data analysis,
data augmentation, CNN model development, transfer learning,
fine-tuning, and model evaluation.

The supplied dataset contains **119 images** across three classes:

-   Bacterial leaf blight: 40 images
-   Brown spot: 40 images
-   Leaf smut: 39 images

> **Dataset note:** The assignment description refers to Leaf Blast and
> 120 images, but the supplied dataset contains Leaf smut and 119
> images. This project uses the labels and images actually provided. The
> discrepancy is documented in the notebook.

## Model Results

The following results were obtained on the held-out test set of 18
images.

  -----------------------------------------------------------------------
  Model               Test accuracy          Test loss     Macro F1-score
  -------------- ------------------ ------------------ ------------------
  Baseline CNN               55.56%             1.0457               0.51

  Augmented CNN              38.89%             1.0986               0.27

  MobileNetV2            **72.22%**         **0.6467**           **0.72**
  transfer                                             
  learning                                             
  (final model)                                        

  Fine-tuned                 61.11%             0.6464               0.60
  MobileNetV2                                          
  -----------------------------------------------------------------------

MobileNetV2 transfer learning achieved the highest test accuracy and
macro F1-score in these experiments.

**Evaluation limitation:** The test set contains only 18 images (six per
class). These results are preliminary and should not be interpreted as
evidence of production readiness. Evaluation on a larger, independent
dataset is needed.

## Repository Structure

``` text
rice-leaf-disease-detection/
├── data/
│   ├── raw/
│   │   └── Data/
│   │       ├── Bacterial leaf blight/
│   │       ├── Brown spot/
│   │       └── Leaf smut/
│   └── .gitkeep
├── models/
│   └── rice_leaf_disease_model.keras
├── notebooks/
│   └── Rice_Leaf_Disease_Detection.ipynb
├── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

The dataset and trained model may be excluded from Git because image
datasets and model files can be large. Keep your local copies backed up
separately.

## Requirements

-   Python 3.11
-   TensorFlow 2.20.0
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   scikit-learn
-   Pillow
-   Jupyter / IPykernel

The original development environment used Ubuntu under WSL2 with an
NVIDIA GPU. GPU acceleration is optional; TensorFlow can also run on a
supported CPU environment.

## Setup

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd rice-leaf-disease-detection
```

Replace `<YOUR_REPOSITORY_URL>` with your Git repository URL.

### 2. Create and activate a virtual environment

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

``` powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the original WSL2 NVIDIA GPU setup, TensorFlow was installed with:

``` bash
pip install "tensorflow[and-cuda]==2.20.0"
```

GPU support depends on the host GPU, drivers, operating system, and
TensorFlow environment. Verify TensorFlow device detection before
assuming GPU acceleration is available.

### 4. Add the dataset

Place the supplied image folders under:

``` text
data/raw/Data/
```

The folder names must match the class names listed above. The dataset is
not included in this repository if it is excluded by `.gitignore`.

### 5. Open the notebook

``` bash
jupyter notebook notebooks/Rice_Leaf_Disease_Detection.ipynb
```

Run the notebook cells in order. The notebook contains the data
analysis, preprocessing, model experiments, evaluation, comparison,
challenges, and conclusion.

## Using the Saved Model

The final model is saved locally as:

``` text
models/rice_leaf_disease_model.keras
```

Load it in Python with:

``` python
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/rice_leaf_disease_model.keras"
)
```

Use the same preprocessing as the notebook: resize images to 224 × 224
with padding, convert pixel values to the \[0, 1\] range, and pass them
to the model. The model includes a rescaling layer that converts inputs
to the range expected by MobileNetV2.

The notebook's prediction demonstration uses a held-out test image. It
is not an independent external-image evaluation.

## Project Limitations

-   The dataset is small (119 images).
-   The assignment description and supplied dataset have a class-label
    discrepancy.
-   The test set is small, so a single image changes overall accuracy by
    about 5.56 percentage points.
-   Brown spot recall was 0.50 in the final model's test report.
-   Model confidence values are softmax outputs and have not been
    calibrated.
-   Larger, diverse, independently collected data is required for
    stronger evaluation.

## License and Dataset

Add the dataset's original license and attribution here if they are
provided by its source. Do not assume that dataset images are free to
redistribute.
