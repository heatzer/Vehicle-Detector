# Vehicle Detection using Histogram of Gradients (HOG) and Machine Learning

This project demonstrates a classic computer vision approach for vehicle detection using the Histogram of Gradients (HOG) feature descriptor combined with several machine learning classifiers. The goal is to accurately distinguish between images that contain vehicles and those that do not.

## 📜 Table of Contents

- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
- [Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation](#-installation)
- [Usage](#-usage)
- [Model Evaluation](#-model-evaluation)
- [Future Work](#-future-work)
- [License](#-license)

## 📝 Project Overview

This repository provides a Python script for a vehicle detection task. The implementation leverages the HOG feature descriptor to capture distinctive features of vehicles in images. These features are then used to train and evaluate three different machine learning models: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest. The performance of these models is compared to identify the most effective classifier for this task.

## 🛠️ Methodology

The project follows these key steps:

1.  **Image Preprocessing**: Images are converted to grayscale and resized to a uniform dimension of $64 \times 64$ pixels to ensure consistency.
2.  **HOG Feature Extraction**: The `skimage.feature.hog` function is used to extract HOG features from the preprocessed images. The HOG descriptor is a powerful tool for capturing shape and appearance information. The key parameters used are:
    - `orientations`: 9
    - `pixels_per_cell`: (8, 8)
    - `cells_per_block`: (2, 2)
    - `block_norm`: 'L2-Hys'
3.  **Dimensionality Reduction**: Principal Component Analysis (PCA) is applied to the high-dimensional HOG features to reduce their dimensionality to 80 principal components. This helps in speeding up the model training process and can improve performance by removing noise.
4.  **Model Training**: The reduced feature set is split into training and testing sets. Three machine learning models are trained on the training data:
    -   **K-Nearest Neighbors (KNN)**
    -   **Support Vector Machine (SVM)** with an RBF kernel
    -   **Random Forest Classifier**
5.  **Model Evaluation**: The trained models are evaluated on the test set using standard classification metrics, including accuracy and a detailed classification report (precision, recall, f1-score). Confusion matrices are also generated for a visual representation of the models' performance.

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python installed on your system. You will also need the following libraries:

-   `numpy`
-   `scikit-image`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`

### Installation

1.  Clone the repository:
    ```sh
    git clone [https://github.com/your-username/vehicle-detection-hog.git](https://github.com/your-username/vehicle-detection-hog.git)
    ```
2.  Install the required packages:
    ```sh
    pip install numpy scikit-image scikit-learn matplotlib seaborn
    ```

## Usage

1.  **Dataset**: This project uses the [Vehicle Detection Image Set](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set) from Kaggle. You need to download the dataset and place the `vehicles` and `non-vehicles` directories in a `data` folder. The script expects the following directory structure:

    ```
    /path/to/your/project
    |-- data/
    |   |-- vehicles/
    |   |   |-- image1.png
    |   |   `-- ...
    |   `-- non-vehicles/
    |       |-- image1.png
    |       `-- ...
    `-- your_script.py
    ```

2.  **Update the base path**: In the script, modify the `base_input_path` variable to point to the location of your `data` folder.

    ```python
    base_input_path = '/path/to/your/data'
    ```

3.  **Run the script**: Execute the Python script to start the training and evaluation process.

    ```sh
    python your_script.py
    ```

## 📊 Model Evaluation

The performance of the trained models on the test set is as follows:

### 🔷 K-Nearest Neighbors (KNN)

-   **Accuracy**: 99.47%
-   **Classification Report**:

|               | precision | recall | f1-score | support |
| :------------ | :-------: | :----: | :------: | :-----: |
| **Non-Vehicle** |   0.99    |  1.00  |   0.99   |  1790   |
| **Vehicle** |   1.00    |  0.99  |   0.99   |  1762   |
| **accuracy** |           |        |   0.99   |  3552   |
| **macro avg** |   0.99    |  0.99  |   0.99   |  3552   |
| **weighted avg**|   0.99    |  0.99  |   0.99   |  3552   |

### 🔶 Support Vector Machine (SVM)

-   **Accuracy**: 99.44%
-   **Classification Report**:

|               | precision | recall | f1-score | support |
| :------------ | :-------: | :----: | :------: | :-----: |
| **Non-Vehicle** |   0.99    |  1.00  |   0.99   |  1790   |
| **Vehicle** |   1.00    |  0.99  |   0.99   |  1762   |
| **accuracy** |           |        |   0.99   |  3552   |
| **macro avg** |   0.99    |  0.99  |   0.99   |  3552   |
| **weighted avg**|   0.99    |  0.99  |   0.99   |  3552   |

### 🌳 Random Forest

-   **Accuracy**: 97.94%
-   **Classification Report**:

|               | precision | recall | f1-score | support |
| :------------ | :-------: | :----: | :------: | :-----: |
| **Non-Vehicle** |   0.97    |  0.99  |   0.98   |  1790   |
| **Vehicle** |   0.99    |  0.97  |   0.98   |  1762   |
| **accuracy** |           |        |   0.98   |  3552   |
| **macro avg** |   0.98    |  0.98  |   0.98   |  3552   |
| **weighted avg**|   0.98    |  0.98  |   0.98   |  3552   |

The KNN and SVM models demonstrate exceptional performance, with both achieving approximately 99.4% accuracy. The Random Forest model also performs well, with an accuracy of around 97.9%.

## 🔮 Future Work

Potential future improvements for this project include:

-   **Hyperparameter Tuning**: Fine-tuning the parameters of the machine learning models (e.g., `n_neighbors` for KNN, `C` and `gamma` for SVM) could lead to even better performance.
-   **Sliding Window Approach**: Implementing a sliding window technique to detect vehicles in larger images at different scales and locations.
-   **Deep Learning Models**: Exploring the use of Convolutional Neural Networks (CNNs) for an end-to-end vehicle detection system, which could potentially yield higher accuracy and robustness.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
