# model_training.py

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from model_creation import create_model  # Import the model creation function

texts = [
    "I love this product", "Horrible experience", "It was okay", "I will buy again", "Not worth the money",
    "not bad", "I am neutral", "I am unsatisfied", "It was bad", "Absolutely amazing",
    "Really great experience", "Terrible purchase", "I feel nothing about it", "Good value for money",
    "I hate it", "It’s perfect", "I regret buying this", "Could have been better", "Fantastic service",
    "It was horrible", "So-so experience", "Perfect fit", "Very disappointed", "I am content with it",
    "It’s fine", "Waste of money", "I adore this", "It’s awful", "Love it", "Worst experience ever",
    "I am satisfied", "I like it", "Not impressed", "I would buy again", "Not recommended", "I’m neutral",
    "Worth every penny", "Not as expected", "So good!", "I’m indifferent", "Would recommend to others",
    "It’s okay", "I dislike this", "I am happy with it", "Horrible product", "Would never buy again",
    "Not worth the hype", "I’m satisfied", "The worst thing ever", "I feel good about it", "It’s awesome",
    "Really bad", "Couldn’t be happier", "Totally worth it", "Terrible quality", "Okay product", 
    "Highly recommend", "Not the best", "Bad investment", "I feel great", "I will not buy again", "I’m in love with this",
    "I am pleased", "Disappointing", "Amazing product", "Good but not great", "Really bad experience", "Just average",
    "Perfect", "I am impressed", "Nothing special", "Horrible quality", "It’s fantastic", "I’m upset",
    "Wonderful purchase", "It’s not bad", "I don’t like it", "It’s okay but could improve", "I am neutral about it",
    "Pretty good", "I’m not a fan", "Wouldn’t buy again", "Very satisfied", "Love it but needs some improvements",
    "It was okay overall", "Good product", "Mediocre product", "I am not happy with this", "Would recommend",
    "Highly satisfied", "It’s terrible", "I’m satisfied with it", "I feel okay", "So bad", "Very pleased with it",
    "Definitely not worth it", "I like it a lot", "Could be better", "A great purchase", "I am satisfied with my choice",
    "Perfect experience", "Very happy with this", "Does the job", "I don’t mind it", "Good but needs some work",
    "It’s a decent product", "Definitely buy again", "Not bad at all", "I am in love with it", "Great product overall",
    "Okay but not amazing", "Love it!", "It’s alright", "I have mixed feelings", "Fantastic experience", "Not worth the price",
    "I wouldn’t buy it again", "Impressive", "It’s just okay", "Wouldn’t recommend", "Not a fan", "It’s fine but not great",
    "Not great", "This product changed my life!", "I am totally disappointed with it.", "Not bad, but I expected more.",
    "It exceeded my expectations!", "I don’t think I’ll use this again.", "It’s alright, not great though.", 
    "Best purchase I’ve ever made!", "The worst decision I’ve ever made.", "It’s a decent product but nothing special.", 
    "Absolutely love this!", "I am done with this product.", "It’s nothing to write home about.", "One of the best things I’ve bought in a while.", 
    "This is absolutely horrible!", "Average at best.", "I would highly recommend this to anyone.", "Would never recommend this to anyone.",
    "I think it’s fine.", "The best product for this price!", "Just doesn’t work as expected.", "It’s okay, but could use some work.",
    "I’m very pleased with it.", "I would not buy this again.", "This product is fine.", "I will definitely buy this again.",
    "I’m not a fan of this product.", "It’s pretty good for the price.", "Amazing experience, loved it!", "Waste of money, avoid this.",
    "It gets the job done.", "Best investment I’ve made in a while.", "Very poor quality.", "It’s acceptable but not impressive.",
    "Absolutely perfect for what I needed.", "I feel completely let down by this product.", "This is as good as it gets.", 
    "I won’t be buying from this brand again.", "It’s okay, but I wouldn’t recommend it.", "Highly recommended!", 
    "Terrible product.", "It’s fine, does the job.", "Wonderful experience.", "Not what I expected at all.", "Good for the price.",
    "I am in love with this product.", "Very disappointed in this.", "Decent, but nothing to get excited about.", "It works just as described.",
    "I wish I had returned it.", "It’s not bad at all.", "Such a great buy!", "It didn’t live up to the hype.", 
    "I’m happy with my purchase.", "Could have been better.", "I regret this purchase.", "It’s a nice product.", "Waste of money.",
    "Very satisfying purchase.", "Okay, but needs some improvement.", "It’s terrible.", "Completely worth it!", "Not for me.", 
    "Could be worse, could be better.", "Great purchase!", "I wouldn’t buy it again.", "I’m very content with it.", "I’m not sure about this one.",
    "Superb quality!", "I’m very dissatisfied with this.", "Just okay.", "Perfect for my needs!", "Bad investment.", "Meh, just fine.",
    "This is the best purchase I’ve made.", "Very poor quality.", "It’s functional.", "I love this!", "Waste of time and money.", 
    "I feel neutral about this.", "It was okay, not amazing.", "Love it, highly recommend.", "Worst decision ever.", "Not exactly what I expected.",
    "Totally worth the investment!", "I’ll never buy from here again.", "I think it’s okay.", "Slightly better than average.", 
    "This is the best purchase I’ve made.", "Very upset with this product.", "I don’t mind it.", "It was a great deal.", "I won’t be recommending this.",
    "Totally satisfied!", "This product is awful.", "Just average, nothing special.", "Best thing I’ve bought in a while.", 
    "Not happy with it.", "This is acceptable.", "So happy with my purchase!", "Very poor quality product.", "I’m okay with it.", 
    "Amazing, worth every penny.", "Definitely a bad buy.", "It’s fine for the price.", "Would buy again.", "Wouldn’t buy again.", 
    "Nice, but not what I expected.", "Completely worth it.", "It’s awful.", "Really happy with it.", "I hate this."
]

labels = [
    1, 0, 2, 1, 0, 2, 2, 0, 0, 1, 1, 0, 2, 1, 0, 1, 0, 2, 0, 1, 2, 1, 0, 1, 2, 0, 1, 1, 0, 1, 2, 1, 0, 2, 
    0, 0, 1, 1, 0, 2, 0, 0, 1, 2, 0, 0, 2, 1, 1, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 
    2, 1, 2, 0, 1, 0, 0, 1, 1, 0, 2, 1, 0, 2, 0, 2, 1, 0, 1, 2, 0, 2, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0, 
    1, 0, 2, 1, 2, 0, 1, 0, 0, 1, 0, 0, 2, 0, 1, 2, 2, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1, 0, 0, 1, 0, 2, 1, 2, 
    1, 0, 1, 0, 1, 0, 2, 1, 1, 1, 0, 2, 0, 1, 2, 1, 2, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 0, 2, 0, 1, 1, 1, 0, 
    1, 2, 0, 0, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 0, 2, 1, 0, 0, 1, 2, 0, 2, 1, 2, 1, 0, 2, 0, 
    1, 2, 1, 0, 1, 0, 2, 1, 2, 0, 2, 1, 1, 2, 1, 0, 2, 0, 0, 1, 0, 1, 2, 0, 2, 1, 1, 2, 1, 1, 
]

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

# Load the model from model_creation.py
model = create_model(input_dim=1000, max_sequence_length=max_sequence_length)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions on the test set
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Output the predicted sentiment classes for the test set
print("Predicted Sentiments (on test data):", predicted_labels)

# Predict sentiment on new data (unseen text)
new_texts = [
    "This product is amazing!",
    "worst product ever",
    "It was just okay, nothing special.",
    "WORST product",
    "I am satisfied with this purchase."
]

# Convert the new texts into sequences and pad them to match the model's input shape
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_sequences, maxlen=max_sequence_length)

# Predict sentiments for the new data
new_predictions = model.predict(new_X)
new_predicted_labels = np.argmax(new_predictions, axis=1)

# Output the predicted sentiment classes for new data
print("\nPredicted Sentiments (on new data):")
for text, label in zip(new_texts, new_predicted_labels):
    sentiment = ['Negative', 'Positive', 'Neutral'][label]  # Map labels back to sentiment categories
    print(f"Text: {text} -> Sentiment: {sentiment}")
