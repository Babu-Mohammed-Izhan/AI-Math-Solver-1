import spacy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def embed_text(text):
    doc = nlp(text)
    
    embedded_tokens = [token.vector for token in doc]
    
    max_length = 2
    padded_embeddings = np.zeros((max_length, 96))
    padded_embeddings[:min(len(embedded_tokens), max_length)] = embedded_tokens[:max_length]
    
    tensor = tf.constant(padded_embeddings.reshape((max_length, 96)))
    
    return tensor


df = pd.read_csv("./math_problems.csv", delimiter=";")


train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

model = keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(None, 2)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def train_model():

    train_labels = train_data['Solution']
    test_labels = test_data['Solution']

    model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), verbose=1)

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save_weights("math_model.h5")

train_model()

def solve_ai(text):

    prediction = model.predict(text)  
    
    return prediction


