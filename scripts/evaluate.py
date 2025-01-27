import os
from tensorflow.keras.models import load_model
from load_data import load_and_preprocess_data

# Define constants
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
SAVE_DIR = 'models'
MODEL_FILENAME = 'wine_model.h5'

def evaluate_model():
    """
    Evaluate a trained Tensorflow model on the test dataset
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_URL)

    # Load the trained model
    model_path = os.path.join(SAVE_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    
    model = load_model(model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model Evaluation:\n Loss: {loss}\n Accuracy: {accuracy}")

    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions (first 5 samples):")
    print(predictions[:5])

if __name__=="__main__":
    evaluate_model()