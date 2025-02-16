# TensorFlow Neural Network Project

This project demonstrates the use of TensorFlow to build, train, and evaluate a neural network for classifying wine samples based on their chemical properties. The dataset used is the Wine dataset from the UCI Machine Learning Repository.

## Project Structure

```
TensorflowNN_Project/
│
├── data/                  # Any datasets (if applicable)
├── models/                # Directory for saving trained models
│   └── wine_model.h5      # Saved model after training
├── scripts/               # Python scripts
│   ├── load_data.py       # Script for loading and preprocessing data
│   ├── model.py           # Script for defining the model
│   ├── train.py           # Script for training the model
│   ├── evaluate.py        # Script for evaluation and prediction
├── README.md              # Project description and instructions
├── environment.yml        # Conda environment file
└── requirements.txt       # Pip dependencies (optional)
```

## Setup Instructions

### Prerequisites
- Python 3.10 or later
- Conda (Miniconda or Anaconda recommended)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fexilgonzo/TensorflowNN_Project.git
   cd TensorflowNN_Project
   ```

2. **Set up the Conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate tensorflow_nn_project
   ```

3. **Run the scripts**:
   - **Train the model**:
     ```bash
     python scripts/train.py
     ```
     This will create and save the trained model in the `models/` directory. If the `models/` directory does not exist, the script will create it automatically.
   - **Evaluate the model**:
     ```bash
     python scripts/evaluate.py
     ```

## Dataset

The dataset is the [Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine) from the UCI Machine Learning Repository. It contains 13 numerical features describing the chemical composition of wines, along with a class label indicating the type of wine.

## Dependencies

The dependencies for this project are specified in the `environment.yml` file. They include:
- TensorFlow
- Scikit-learn
- Pandas

To install dependencies manually using `pip`, use the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Notes

- The `models/` directory is excluded from version control, as the trained model (`wine_model.h5`) can be regenerated by running `train.py`.
- The `train.py` script will automatically create the `models/` directory if it does not exist.

## Project Workflow

1. **Load and preprocess data**: Use `load_data.py` to load the dataset and split it into training and testing sets.
2. **Define the model**: Use `model.py` to define a neural network architecture.
3. **Train the model**: Use `train.py` to train the model and save the best model.
4. **Evaluate the model**: Use `evaluate.py` to evaluate the trained model and make predictions.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
