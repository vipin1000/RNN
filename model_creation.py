# model_creation.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(input_dim, max_sequence_length):
    """
    Creates and returns a compiled RNN model for sentiment analysis.
    Args:
        input_dim (int): The number of words in the tokenizer's vocabulary.
        max_sequence_length (int): The length of input sequences.

    Returns:
        model: A compiled Keras Sequential model.
    """

    model = Sequential()

    # Embedding Layer: Converts input tokens to dense vectors
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_sequence_length))

    # RNN Layer: Simple RNN (could also use LSTM/GRU)
    model.add(SimpleRNN(64, return_sequences=False))

    # Optional Dropout Layer: Prevents overfitting
    model.add(Dropout(0.5))

    # Output Layer: Predict sentiment classes (1, 0, or 2)
    model.add(Dense(3, activation='softmax'))  # 3 classes: positive, negative, neutral

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
