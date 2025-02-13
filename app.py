
import numpy as np
from tensorflow.keras.models import Sequential                                         
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout         
from tensorflow.keras.preprocessing.text import Tokenizer                              
from tensorflow.keras.preprocessing.sequence import pad_sequences                       
from tensorflow.keras.optimizers import Adam                                           
from sklearn.model_selection import train_test_split

# Sample data (sentences and sentiment labels)
texts = [
    "I love this product", 
    "Horrible experience", 
    "It was okay", 
    "I will buy again", 
    "Not worth the money",
    "not bad",
    "I am neutral",
    "I am unsatisfied",
    "It was bad",
    "Absolutely amazing",
    "Really great experience",
    "Terrible purchase",
    "I feel nothing about it",
    "Good value for money",
    "I hate it",
    "It’s perfect",
    "I regret buying this",
    "Could have been better",
    "Fantastic service",
    "It was horrible",
    "So-so experience",
    "Perfect fit",
    "Very disappointed",
    "I am content with it",
    "It’s fine",
    "Waste of money",
    "I adore this",
    "It’s awful",
    "Love it",
    "Worst experience ever",
    "I am satisfied",
    "I like it",
    "Not impressed",
    "I would buy again",
    "Not recommended",
    "I’m neutral",
    "Worth every penny",
    "Not as expected",
    "So good!",
    "I’m indifferent",
    "Would recommend to others",
    "It’s okay",
    "I dislike this",
    "I am happy with it",
    "Horrible product",
    "Would never buy again",
    "Not worth the hype",
    "I’m satisfied",
    "The worst thing ever",
    "I feel good about it",
    "It’s awesome",
    "Really bad",
    "Couldn’t be happier",
    "Totally worth it",
    "Terrible quality",
    "Okay product",
    "Highly recommend",
    "Not the best",
    "Bad investment",
    "I feel great",
    "I will not buy again",
    "I’m in love with this",
    "I am pleased",
    "Disappointing",
    "Amazing product",
    "Good but not great",
    "Really bad experience",
    "Just average",
    "Perfect",
    "I am impressed",
    "Nothing special",
    "Horrible quality",
    "It’s fantastic",
    "I’m upset",
    "Wonderful purchase",
    "It’s not bad",
    "I don’t like it",
    "It’s okay but could improve",
    "I am neutral about it",
    "Pretty good",
    "I’m not a fan",
    "Wouldn’t buy again",
    "Very satisfied",
    "Love it but needs some improvements",
    "It was okay overall",
    "Good product",
    "Mediocre product",
    "I am not happy with this",
    "Would recommend",
    "Highly satisfied",
    "It’s terrible",
    "I’m satisfied with it",
    "I feel okay",
    "So bad",
    "Very pleased with it",
    "Definitely not worth it",
    "I like it a lot",
    "Could be better",
    "A great purchase",
    "I am satisfied with my choice",
    "Perfect experience",
    "Very happy with this",
    "Does the job",
    "I don’t mind it",
    "Good but needs some work",
    "It’s a decent product",
    "Definitely buy again",
    "Not bad at all",
    "I am in love with it",
    "Great product overall",
    "Okay but not amazing",
    "Love it!",
    "It’s alright",
    "I have mixed feelings",
    "Fantastic experience",
    "Not worth the price",
    "I wouldn’t buy it again",
    "Impressive",
    "It’s just okay",
    "Wouldn’t recommend",
    "Not a fan",
    "It’s fine but not great",
    "Not great"
]
# Labels for sentiment: 1 = Positive, 0 = Negative, 2 = Neutral
labels = [1, 0, 2, 1, 0, 2, 2, 0, 0, 1, 1, 0, 2, 1, 0, 1, 0, 2, 0, 1, 2,
          1, 0, 1, 1, 0, 1, 2, 0, 0, 1, 0, 1, 1, 2, 0, 1, 0, 0, 2, 1, 1, 
          0, 1, 0, 2, 0, 0, 2, 1, 2, 0, 1, 1, 0, 2, 1, 0, 2, 1, 2, 1, 0, 
          1, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 1, 0, 0, 1, 1, 0, 2,0,2,1,2,1, 
          0, 1, 2, 0, 0, 1, 2, 1, 0, 1, 1, 0, 2, 0, 1, 0, 0, 2, 0, 1,1,2,1,2,1,2,1,0,0,1,2,1,0,0]

# Tokenize the text data (convert text into sequences of integers)
tokenizer = Tokenizer(num_words=1000)  # Only keep the top 1000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to make them of equal length
max_sequence_length = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels into a numpy array
y = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the RNN Model
model = Sequential()

# Embedding Layer: Converts input tokens to dense vectors
model.add(Embedding(input_dim=1000, output_dim=128, input_length=max_sequence_length))

# RNN Layer: Simple RNN (could also use LSTM/GRU)
model.add(SimpleRNN(64, return_sequences=False))

# Optional Dropout Layer: Prevents overfitting
model.add(Dropout(0.5))

# Output Layer: Predict sentiment classes (1, 0, or 2)
model.add(Dense(3, activation='softmax'))  # 3 classes: positive, negative, neutral

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Output the predicted sentiment classes
print("Predicted Sentiments:", predicted_labels)


