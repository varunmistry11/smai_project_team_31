# SMAI Project Team 31
# AUTOMATED FOUL IDENTIFICATION IN SOCCER USING MULTI-VIEW VIDEO ANALYSIS

This repository contains implementations for the project of Automated Foul Identification in Soccer using Multi-View Video Analysis.

## Repository Structure

```
smai_project_team_31/
├── dataset/
│   └── download_data.py   # Script to download the dataset
├── cnn/
│   └── ...                # CNN model implementation
└── transformer/
    ├── model/             # Transformer model training and inference
    │   └── ...
    └── ui/                # User interface for viewing predictions
        └── ...
```

## Dataset

The `dataset` folder contains a script for downloading the dataset:

```bash
python dataset/download_data.py
```
After downloading dataset, extract the `train`, `val` and `test` folders and place them in `dataset` directory. 

## Models

Both CNN and Transformer models have been trained and are available for download:

- [CNN Model Download Link](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/varun_mistry_students_iiit_ac_in/EdqICQ2XrLFBoRzLt_KJE7kBzqaZcvzlFJ7Qew0apPpIag?e=3NIaZc) best_mvfoul_model.pth
- [Transformer Model Download Link](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/varun_mistry_students_iiit_ac_in/EfVB-HEHu8pIoB0QYjIf9UMBjlzRihXyTCmgZlnWC3xhHQ?e=g5l7aJ) 14_model.pth.tar

Please download the pre-trained models and place them in their respective model folders as follows.
- Place `best_mvfoul_model.pth` in `cnn` folder
- Place `14_model.pth.tar` in `transformer/model` and `transformer/ui` folder

## Transformer Model Instructions

### Setup Instructions

Run the following commands to set up the environment:

```bash
cd transformer/model
# Create and activate conda environment
conda create -n vars python=3.9
conda activate vars

# Install PyTorch with CUDA
# Visit https://pytorch.org/get-started/locally/ for installation command based on your system

# Install dependencies
pip install SoccerNet
pip install -r requirements.txt
pip install pyav
```

### Training

To start the training from scratch:

```bash
cd transformer/model
python main.py --path "path/to/dataset"
```

### Using Pre-trained Weights

```bash
cd transformer/model
python main.py --pooling_type "attention" --start_frame 63 --end_frame 87 --fps 17 --path "path/to/dataset" --pre_model "mvit_v2_s" --path_to_model_weights "14_model.pth.tar"
```

## Transformer UI Interface

### Setup Instructions

```bash
cd transformer/ui
# Create and activate conda environment
conda create -n vars python=3.9
conda activate vars

# Install dependencies
pip install -r requirements.txt
pip install av
```

### Running the UI

Once the environment is ready, run the annotation tool for camera shots and replays:

```bash
cd transformer/ui
python main.py
```

Then select one or several clips in the folder "Dataset". 

## File Descriptions

### Dataset Files

- `download_data.py`: Script to download and extract the required dataset for training and evaluation.

### CNN Model Files

- `model.py`: Defines the CNN architecture used for video analysis.
- `train.py`: Training script for the CNN model with data loading, optimization, and evaluation.
- `inference.py`: Script to run inference using the trained CNN model.
- `utils.py`: Helper functions for data processing, transformations, and other utilities.

### Transformer Model Files

- `main.py`: Main entry point for training and evaluation of the transformer model.
- `model.py`: Implementation of the transformer-based architecture with attention mechanisms.
- `dataset.py`: Dataset loader and preprocessing functions specific to the transformer model.
- `utils.py`: Utility functions for training, logging, and evaluation metrics.
- `config.py`: Configuration parameters for the transformer model.

### Transformer UI Files

- `main.py`: Entry point for the UI application that displays video analysis results.
- `ui_utils.py`: Utility functions for the user interface.
- `visualization.py`: Functions for visualizing model predictions and attention maps.
- `inference.py`: Real-time inference module for the UI to process video input.