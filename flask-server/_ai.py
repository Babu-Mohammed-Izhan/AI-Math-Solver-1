import spacy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def embed_text(text):
    doc = nlp(text)
    
    embedded_tokens = [token.vector for token in doc]
    
    max_length = 50
    padded_embeddings = np.zeros((max_length, 300))
    padded_embeddings[:min(len(embedded_tokens), max_length)] = embedded_tokens[:max_length]
    
    tensor = tf.constant(padded_embeddings.reshape((1, max_length, 300)))
    
    return tensor


df = pd.read_csv("math_problems.csv")


train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)


model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(50, 300)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1)
])


model.compile(loss="mse", optimizer="adam")


def train_model():

    X_train = np.array([embed_text(text) for text in train_df["text"]])
    y_train = np.array(train_df["answer"])
    

    X_val = np.array([embed_text(text) for text in val_df["text"]])
    y_val = np.array(val_df["answer"])
    

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    

    model.save_weights("math_model.h5")


train_model()


def ai(text):

    tensor = embed_text(text)
    
    answer = model.predict(tensor)[0][0]
    
    return answer


