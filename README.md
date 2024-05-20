# tree_classifier
# Mango Classifier

Welcome to the Mango Classifier repository! This project aims to classify different species of mango trees based on the morphological features of their leaves using a neural network.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Limitations](#limitations)
- [Contributing](#contributing)

## Project Overview

The goal of this project is to build a robust neural network model that can accurately classify mango tree species from their mature leaves. The model leverages deep learning techniques to capture complex patterns in leaf morphology.

## Dataset

The dataset used in this project is stored in an Excel file named `tree_species_data.xlsx`. It contains:
- **120 samples**
- **4 columns**: the first column is the species label, and the remaining three are feature measurements (ratio, angle, length of petiole).

### Data Preprocessing

- **Label Encoding**: Species names are encoded as integers using `LabelEncoder`.
- **One-Hot Encoding**: Encoded labels are converted to one-hot vectors for model training.
- **Normalization**: Feature values are normalized to the range [0, 1].

### Splitting

- **Training Set**: 80% of the data
- **Validation Set**: 20% of the training set
- **Testing Set**: 20% of the total data

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow numpy pandas scikit-learn
```

## Usage

### Training the Model

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/mango_classifier.git
    cd mango_classifier
    ```

2. **Run the training script**:
    ```bash
    python train.py
    ```

### Evaluating the Model

1. **Run the evaluation script**:
    ```bash
    python evaluate.py
    ```

### Prediction

1. **Run the prediction script**:
    ```bash
    python predict.py --input <path_to_input_data>
    ```

## Model Architecture

The neural network model comprises:

- **Input Layer**: Normalized feature data
- **Hidden Layers**: Seven dense layers with LeakyReLU activations (neurons: 1024, 512, 512, 256, 128, 128, 32)
- **Output Layer**: Softmax activation with 3 neurons (for 3 species)

### Training Parameters

- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 20
- **Epochs**: 25
- **Cross-Validation**: 5-Fold

## Results

The model achieved high accuracy in classifying the three species of mango trees. Key findings include:

- **Validation Accuracy**: Consistent accuracy across folds due to robust cross-validation.
- **Effective Hyperparameters**: The choice of network architecture, optimizer, and activation functions significantly contributed to the model's performance.

## Limitations

- **Specific to Mature Leaves**: The model is designed for mature leaves and may not perform well with immature leaves.
- **Small Dataset**: Limited to 120 samples, which might affect the generalizability.
- **Species Diversity**: The model is trained on a limited number of species.
- **Environmental Factors**: Performance may vary with environmental conditions.

## Contributing

We welcome contributions to improve the Mango Classifier. If you would like to contribute, please fork the repository and create a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a pull request

---

Thank you for checking out the Mango Classifier project! If you have any questions or suggestions, feel free to open an issue or contact us.
