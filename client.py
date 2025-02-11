import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os, sys

# Global variable for number of clients
NUM_CLIENTS = 10 

"""
============================================================================
          Dataset Operations and Partitioning for Clients
============================================================================
"""

# 1) Loading and viewing the dataset
data = pd.read_csv('./Dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

# 2) Cleaning the dataset
data['instruction'] = data['instruction'].str.replace(r"[^\w\s]", "", regex=True)

# 3) Shuffling the dataset by controlling the random state and resetting the index
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 4) Partitioning the dataset to different clients into equal number of rows
data_splits = np.array_split(data, NUM_CLIENTS)

# Client-specific data loading
def load_client_data(client_id):
    # 1) Get the data for the current client
    client_data = data_splits[client_id]
    
    # 2) Tokenizing the instruction where the num_words is the maximum number of words to keep based on word frequency in the vocabulary and the rest will be replaced by unknown,the vocabulary is getting generated with the help of fit_on_texts and most frequent word is assigned the lowest number and afterwords the texts are converted to sequences and padded to make them of same length i.e maxlen for batch processing
    tokenizer = Tokenizer(num_words=1300)
    X = client_data['instruction'].values
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_processed = pad_sequences(sequences, maxlen=25)

    # 3) Encoding the target variable and assigning the classes to the respective labels alphabaetically
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(client_data['category'].values)
    
    # 4) Perform train-test split by testing on 40% of the data and controlling randomness
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    
    # 5) Print client-specific vocabulary and label encoding
    # print(f"Client {client_id} vocabulary: {tokenizer.word_index}\n")
    print(f"Client {client_id} label mapping: {dict(enumerate(label_encoder.classes_))}\n")
    print(f"Input matrix shape: {X_processed.shape} and output matrix shape: {y.shape}\n")
    
    return X_train, X_test, y_train, y_test, tokenizer, label_encoder

# Create model with input parameters as number of rows and number of classes
def create_model(input_shape, num_classes):
    '''
    1) Create a simple sequential model with embedding layer, global average pooling, dense layer with relu activation and output layer with softmax activation:
        a) Embedding layer: This layer will take 25 length sequence of words and convert them into 64 dimensional vector(64 features representing the word) and the output shape will be (25,64) where 25 is the maximum length of the input sentence that we are giving to the model and the output will be a 25*64 matrix as every word of that sentence will be vectorized into 64 characteristics.The 1300 is the vocabulary size which is the number of unique words in the dataset that the model needs to expect.
        b) Global Average Pooling: The input given to this layer will be the 25*64 matrix and the output will be a 64 dimensional vector of (64,) shape which is the average of the features(64 averaged features) which helps to tells you how important each feature is on average in the entire sequence i.e it summarizes the sentence.
        c) Dense layer: This layer has 64 neurons and each neuron receives all 64 feature values as inputs.Every neuron learns to detect or respond to a specific combination of those features.Each neuron applies : z = (w1 * x1) + (w2 * x2) + ... + (w64 * x64) + b where w1,w2,...,w64 are the weights and x1,x2,...,x64 are the input features and b is the bias.Then it is passed through ReLU activation function a non-linear function: f(z) = max(0,z) which helps the model to ignore irrelevant patterns (negative values are set to 0) and focuse on important patterns (positive values are kept as they are).
        d) Dense layer: This layer has 11 neurons which is the number of classes in the dataset in this case now it combines the 64 features into a single value (or logit) for each class.For neuron j in the output layer: z_j = (w1_j * x1) + (w2_j * x2) + ... + (w64_j * x64) + b_j where w1_j,w2_j,...,w64_j are the weights and x1,x2,...,x64 are the input features and b_j is the bias.Each neuron in this dense layer corresponds to one of the 11 classes and then the softmax activation takes the 11 logits and converts them into probabilities(11 probability outcomes i.e (11,)) that sum to 1.This way, the model gives a probability for each class, and the class with the highest probability is the prediction.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(1300, 64, input_length = input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 2) Compiled the model with adam optimizer, sparse categorical crossentropy loss and accuracy as the metric.Adam (Adaptive Moment Estimation) adapts the learning rate during training i.e how big or small each correction should be by keeping track of momentum.Sparse Categorical Crossentropy is used when the classes are mutually exclusive (each entry is in exactly one class) and it measures how much far is the predicted probabilities are from the correct label: Loss = -log(P(correct class)) .The loss function measures how well the model is performing on its training data and then the optimizer tries to minimize this loss function.The metric is used to judge the performance of the model: Accuracy = (Number of correct predictions / Total predictions) * 100
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.build(input_shape=(None, input_shape))
    print(f"Model summary for client {client_id} is:")
    print(model.summary())
    
    return model

# Prediciting the model with custom input
def predict_custom_input(client, custom_input):
    # Preprocess the custom input
    sequence = client.tokenizer.texts_to_sequences([custom_input])
    padded_sequence = pad_sequences(sequence, maxlen=25)
    
    # Perform prediction
    probabilities = client.model.predict(padded_sequence)
    predicted_class_index = np.argmax(probabilities)
    
    # Decode the predicted class
    predicted_class = client.label_encoder.inverse_transform([predicted_class_index])[0]
    
    print(f"\nCustom input: '{custom_input}'")
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities[0]}\n")

class BitextClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X_train, self.X_test, self.y_train, self.y_test, self.tokenizer, self.label_encoder = load_client_data(client_id)
        self.model = create_model(self.X_train.shape[1], len(np.unique(self.y_train)))
        self.local_round = 0 # Local round counter

        # Creating directory if it doesn't exist
        if not os.path.exists('ClientModel'):
            os.makedirs('ClientModel')
        self.model_path = os.path.join('ClientModel', f'model_client_{client_id}.keras')

    def get_parameters(self, config):
        #This function returns the weights of the model and config is used if we need layer specific weights.
        weights = self.model.get_weights()
        return weights

    def fit(self, parameters, config):
        
        # Increment the round counter
        self.local_round += 1

        print(f"\n######################## STEP 1 for Round {self.local_round}: Local Model Training for Client {self.client_id} ######################\n")
        
        # 1) Set the model’s weights to the values passed as parameters.
        self.model.set_weights(parameters)
        print("\nModel Weights Before training:")
        # for i, weight in enumerate(self.model.get_weights()):
        #     print(f"\nWeight {i+1}: {weight.shape}\n{weight}")

        # 2) Train the model for 5 epochs(rounds) with a batch size of 64 and a validation split of 20%.The training data is used to adjust the weights of the model and the validation data is used to evaluate the model after each epoch to see how well it is generalizing to unseen data.Batch size controls how many samples the model processes before updating weights. A smaller batch size means that the model is updated more often and the learning has more variance. A larger batch size means that the model is updated less often and the learning has less variance.Verbose is used for displaying the training process.
        print(f"\nStarting local training for client {self.client_id}...\n")
        history = self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
        try:
            self.model.save(self.model_path)
            print(f"\nModel for client {self.client_id} saved successfully at {self.model_path}\n")
        except Exception as e:
            print(f"\nError saving model for client {self.client_id}: {str(e)}")
        
        # 3) Accessing the loss and accuracy of the model after training
        loss = history.history['loss'][-1]
        accuracy = history.history['accuracy'][-1]

        print("\nModel Weights After Training:")
        # for i, weight in enumerate(self.model.get_weights()):
        #     print(f"\nWeight {i+1}: {weight.shape}\n{weight}")

        print("\n============= Local Training Completed =============\n")
        print("\n============= Sending updated local model to server =============\n")

        return self.model.get_weights(), len(self.X_train), {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        # 1) Set the model’s weights to the values passed as parameters.
        self.model.set_weights(parameters)

        print(f"\n################## STEP 2 for Round {self.local_round}: Global Model Testing and updating the client model for client {self.client_id} ######################\n")

        # 2) Evaluate the model on the test data and return the loss and accuracy
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f"\nClient {self.client_id} - Evaluation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
        print(f"\n============= Client {self.client_id} Testing Completed =============\n")
        print(f"\n============= Sending evaluation results to server =============\n")

        return loss, len(self.X_test), {"accuracy": accuracy}

# Start client (run this with client_id=0 to NUM_CLIENTS-1 in separate processes)
if __name__ == "__main__":
    client_id = int(sys.argv[1])
    print(f"\n*******************************Client {client_id} is running*****************************\n\n")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BitextClient(client_id)
    )
    print(f"\n============= Client {client_id} Completed with updating with the final global model =============\n")


#Testing purposes
# if __name__ == "__main__":
#     client_id = int(sys.argv[1])
#     print(f"\n******************************Client {client_id} is running********************************\n\n")

#     # Create a client instance
#     client = BitextClient(client_id)
    
#     # Simulate training
#     initial_weights = client.get_parameters(config={})
#     new_weights, num_examples, metrics = client.fit(initial_weights, config={})

#     # Simulate evaluation
#     loss, num_eval_examples = client.evaluate(new_weights, config={})

#     custom_input = "when you deliver my parcel"
#     predict_custom_input(client, custom_input)