from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate

def build_model(input_shape):
    """
    Build a Tensorflow neural network model

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: Compiled Keras model.
    """
    # Define the input layer
    input_ = Input(shape=input_shape, name='input_layer')

    # Hidden layers
    hidden1 = Dense(30, activation='relu', name='dense_1')(input_)
    hidden2 = Dense(30, activation='relu', name='dense_2')(hidden1)
    hidden3 = Dense(30, activation='relu', name='dense_3')(hidden2)

    # Concatenate input and final hidden layer
    concat = concatenate([input_, hidden3], name='concat_layer')

    # Output layer
    output = Dense(1, name='output_layer')(concat)

    # Create the model
    model = Model(inputs=[input_], outputs=[output])

    # Compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Usage
if __name__=="__main__":
    input_shape = (13,)
    model = build_model(input_shape)
    model.summary()