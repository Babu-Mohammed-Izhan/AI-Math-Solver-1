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

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['Problem'])
train_sequences = tokenizer.texts_to_sequences(train_data['Problem'])
test_sequences = tokenizer.texts_to_sequences(test_data['Problem'])


max_len = 100

model = keras.Sequential([
    keras.layers.Embedding(len(tokenizer.word_index)+1, 64, input_length=max_len),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def train_model():

    max_len = 100
    train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')


    train_labels = train_data['Solution']
    test_labels = test_data['Solution']

    model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels), verbose=1)

    test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save_weights("math_model.h5")

def solve_ai(text):

    train_model()

    example_seq = tokenizer.texts_to_sequences(text)
    example_padded = tf.keras.preprocessing.sequence.pad_sequences(example_seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(example_padded)  
    
    return prediction


