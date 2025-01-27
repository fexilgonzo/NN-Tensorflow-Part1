import os
from load_data import load_and_preprocess_data
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Define constants
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
SAVE_DIR = 'models'
MODEL_FILENAME = 'wine_model.h5'

def train_model():
    """
    Train a Tensorflow neural network on the wine dataset
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_URL)

    # Build the model
    input_shape = (X_train.shape[1],)
    model = build_model(input_shape)

    # Ensure save directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Define a checkpoint callback
    checkpoint = ModelCheckpoint(filepath=os.path.join(SAVE_DIR, MODEL_FILENAME), 
                                 save_best_only=True, monitor='val_loss', mode='min')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint]
    )

    # Print training completion message
    print(f"Training complete. Model saved to {os.path.join(SAVE_DIR, MODEL_FILENAME)}")

if __name__=="__main__":
    train_model()