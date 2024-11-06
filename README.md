# Quality Assurance using Computer Vision and Deep Learning

This project utilizes computer vision and deep learning to perform automated quality assurance on products. By using convolutional neural networks, this system classifies products as either defective or non-defective, based on image data. The project is built with TensorFlow and leverages a pre-trained Xception model fine-tuned for this task.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Overview
The goal of this project is to create a computer vision system capable of detecting defects in products, which can be used in quality assurance processes. The system classifies images of products into two categories:
- **Defective**
- **Non-defective**

The project uses a pre-trained Xception model with additional layers for classification. The model is trained on a dataset of product images with labels indicating whether each product is defective or non-defective.

## Dataset
The dataset consists of images organized into two directories:
- `train/defective`: Images of defective products.
- `train/non_defective`: Images of non-defective products.
- `test`: Images for testing the model.

Each image is resized to `(300, 300)` pixels and normalized.

## Model Architecture
This project uses a deep learning model based on the **Xception** architecture, a powerful CNN model pre-trained on the ImageNet dataset. The architecture is fine-tuned by adding a Global Average Pooling layer and two Dense layers:
1. A Dense layer with ReLU activation for feature extraction.
2. A Dense layer with Softmax activation for binary classification.

### Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 20
- Batch Size: 64

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quality-assurance-computer-vision.git
   cd quality-assurance-computer-vision
   ```
2. Install the required dependencies:
   ```bash
   pip install tensorflow tensorflow-macos tensorflow-metal matplotlib scikit-learn
   ```

3. Set up the dataset structure:
   ```
   data/
   ├── train/
   │   ├── defective/
   │   └── non_defective/
   └── test/
       ├── defective/
       └── non_defective/
   ```

## Usage
1. **Train the Model**:
   Run the `Quality Assurance using Computer Vision and Deep Learning.ipynb` notebook to train the model on the provided dataset.

2. **Evaluate the Model**:
   After training, evaluate the model's performance using the test dataset. A confusion matrix is generated to visualize the model's classification accuracy.

3. **Generate Predictions**:
   The notebook includes code to generate predictions on new images, which can be used to classify unseen products as defective or non-defective.

## Results
The model achieves high accuracy on the test set, effectively distinguishing between defective and non-defective products. The confusion matrix provides insights into the model's performance and any areas where misclassifications may occur.

## Future Work
- Explore additional model architectures (e.g., ResNet, MobileNet) to further improve accuracy.
- Implement a real-time defect detection pipeline using OpenCV.
- Experiment with other datasets to generalize the model to different types of products.

## License
This project is licensed under the MIT License.
