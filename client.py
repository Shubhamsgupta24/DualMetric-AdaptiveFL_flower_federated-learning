import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Global variable for number of clients
NUM_CLIENTS = 2

# Load and preprocess dataset
data = pd.read_csv('../Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

# Preprocessing
tokenizer = Tokenizer(num_words=1000)
label_encoder = LabelEncoder()

# Preprocess all data at once
X = data['instruction'].values
y = label_encoder.fit_transform(data['intent'].values)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_processed = pad_sequences(sequences, maxlen=50)

# Client-specific data partitioning
def load_client_data(client_id):
    # Split data into NUM_CLIENTS parts
    client_data = np.array_split(X_processed, NUM_CLIENTS)[client_id]
    client_labels = np.array_split(y, NUM_CLIENTS)[client_id]
    return client_data, client_labels

# Create model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(1000, 64, input_length=input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class BitextClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X, self.y = load_client_data(client_id)
        self.model = create_model(self.X.shape[1], len(np.unique(y)))
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X, self.y, epochs=5, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {"accuracy": accuracy}

# Start client (run this with client_id=0 to NUM_CLIENTS-1 in separate processes)
if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])
    print(f"Client {client_id} is running")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BitextClient(client_id)
    )
