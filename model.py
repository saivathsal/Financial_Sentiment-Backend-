import json
import spacy
import tensorflow as tf
from tensorflow.keras import layers
from config import model_path,config_path,vocab_path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

le = LabelEncoder()
class_labels = ['positive', 'neutral', 'negative'] 
le.fit(class_labels)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class Attention(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(x * weights, axis=1)
        return context
    
model = load_model(model_path, custom_objects={'Attention': Attention})

def lemmatize(text):
    return " ".join([t.lemma_ for t in nlp(text.lower()) if not t.is_punct and not t.is_space])

with open(config_path, "r") as file:
    config = json.load(file)

import numpy as np
vocab = np.load(vocab_path, allow_pickle=True)

vectorizer = layers.TextVectorization(
    max_tokens=config["max_vocab"],
    output_sequence_length=config["max_len"],
    standardize=None,
    split="whitespace",
    vocabulary=vocab 
)

def predict(text):
    clean = lemmatize(text)
    vectorized = vectorizer(tf.convert_to_tensor([clean]))
    pred = model.predict(vectorized, verbose=0)
    pred_class_idx = tf.argmax(pred[0]).numpy()
    label = le.classes_[pred_class_idx]
    confidence = float(pred[0][pred_class_idx])
    return  str(label), confidence
