# Fine-Tune-VideoMAE on INCLUDE ISL Dataset
# Video Classification using VideoMAE

This project demonstrates how to perform video classification using the VideoMAE model from the Hugging Face Transformers library. The notebook covers data loading, preprocessing, model definition, training, and evaluation.

## Overview

The notebook utilizes the VideoMAE (Masked Autoencoder for Video) model, a self-supervised pre-training approach for video understanding. It fine-tunes a pre-trained VideoMAE model on a video dataset for classification.

## Notebook Content

The notebook `videomae.ipynb` includes the following steps:

1. **Import Libraries**: Imports necessary libraries, including NumPy, Pandas, PyTorch, Transformers, and PyAV.
2. **Data Loading and Preprocessing**:
    *   Defines the dataset folder path.
    *   Reads video file paths and extracts labels from the folder structure.
    *   Creates a Pandas DataFrame to store video paths and their corresponding labels.
    *   Splits the data into training, testing, and validation sets.
3. **Install Dependencies**: Installs required packages like `pytorchvideo`, `transformers`, `evaluate`, `accelerate`, and `av`.
4. **Helper Functions**: Defines helper functions:
    *   `read_video_pyav`: Decodes video frames using PyAV.
    *   `sample_frame_indices`: Samples frame indices from a video.
    *   `get_image_processor_inputs`: Preprocesses video frames using `AutoImageProcessor`.
5. **Dataset Class**: Creates a `VideoClassificationDataset` class that inherits from `torch.utils.data.Dataset` to handle video data loading and preprocessing.
6. **Load Pre-trained Model and Image Processor**: Loads the pre-trained VideoMAE model and its corresponding image processor from Hugging Face.
7. **Data Loaders**: Creates `DataLoader` instances for training, testing, and validation datasets.
8. **Video Classifier Model**: Defines a `VideoClassifier` class that fine-tunes the pre-trained VideoMAE model by adding a classification head.
9. **Training Setup**: Initializes the model, optimizer (AdamW), and loss function (CrossEntropyLoss).
10. **Training Loop**: Implements the training process, including forward and backward passes, loss calculation, and optimization. It also includes evaluation on the validation set after each epoch and saves the best model.
11. **Evaluation Function**: Defines an `evaluate` function to calculate the loss, accuracy, and F1-score on a given dataset.

## Usage

To use this notebook:

1. **Install Dependencies**: Make sure you have the required libraries installed (`pytorchvideo`, `transformers`, `evaluate`, `accelerate`, `av`). You can run the `pip install` commands in the notebook.
2. **Prepare your dataset**: Organize your video files in folders, where the folder name represents the class label. Update the `folder_path` variable in the notebook to point to your dataset directory.
3. **Run the notebook**: Execute the cells in the `videomae.ipynb` notebook sequentially.

This notebook provides a comprehensive guide to fine-tuning the VideoMAE model for video classification tasks.
