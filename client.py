import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
import os, sys
import json
import pickle
import shutil
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

'''
CAUTION:
1) Ensure that the dataset is present in the Dataset folder.
2) NUM_CLIENTS will be according to the number of datasets present in the Train folder which are created using different cases in data_prep_and_viz.py.
'''

# Global variables
NUM_CLIENTS = 6
TOKENIZER_DIR = "./Tokenizer"
LABEL_ENCODER_DIR = "./LabelEncoder"
CLIENT_MODEL_DIR = "./ClientModel"

# Setting print options for better readability where precision is the decimal digits to be printed and threshold is the number of array elements which triggers summarization in a numpy array.
np.set_printoptions(precision=5, threshold=50)

# Clear the Tokenizer,Label Encoder and ClientModel directories before savint them
if os.path.exists(TOKENIZER_DIR):
    shutil.rmtree(TOKENIZER_DIR)  # Remove all files
os.makedirs(TOKENIZER_DIR)  # Recreate the directory

if os.path.exists(LABEL_ENCODER_DIR):
    shutil.rmtree(LABEL_ENCODER_DIR) # Remove all files
os.makedirs(LABEL_ENCODER_DIR) # Recreate the directory

if os.path.exists(CLIENT_MODEL_DIR):
    shutil.rmtree(CLIENT_MODEL_DIR) # Remove all files
os.makedirs(CLIENT_MODEL_DIR) # Recreate the directory

# Client-specific data loading
def load_client_data(client_id):
    # 1) Get train and test data for the current client
    train_data = pd.read_csv(f'./Dataset/Train/train_data_client{client_id}.csv')
    test_data = pd.read_csv('./Dataset/Test/global_test_set.csv')
    
    # 2) Tokenizing the instruction where the num_words is the maximum number of words to keep based on word frequency in the vocabulary and the rest will be replaced by unknown,the vocabulary is getting generated with the help of fit_on_texts and most frequent word is assigned the lowest number and afterwords the texts are converted to sequences and padded to make them of same length i.e maxlen for batch processing
    tokenizer = Tokenizer(num_words=800)
    X_train = train_data['instruction'].values
    X_test = test_data['instruction'].values
    
    tokenizer.fit_on_texts(X_train)
    
    X_train_processed = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=25)

    # In the case of test data,words seen in training are converted to their respective integer IDs whereas the words not seen in training are ignored(treated as out-of-vocabulary).
    X_test_processed = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=25)
    
    # 3) Encoding the target variable and assigning the classes to the respective labels alphabaetically
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data["category"].values)
    
    # 4) Encoding the target variable for the test data and assigning the known classes from the fit to the respective labels alphabaetically whereas the unseen classes are assigned -1
    y_test = []
    for category in test_data["category"]:
        if category in label_encoder.classes_:
            y_test.append(label_encoder.transform([category])[0])
        else:
            y_test.append(-1)  # Assign -1 for unseen categories

    y_test = np.array(y_test)  # Convert to NumPy array
    
    # 5) Saving tokenizer and label encoder for future use
    tokenizer_path = os.path.join(TOKENIZER_DIR, f'tokenizer_client{client_id}.json')
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.to_json(), f)
    print(f"Tokenizer saved at: {tokenizer_path}\n")

    label_encoder_path = os.path.join(LABEL_ENCODER_DIR, f'label_encoder_client{client_id}.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label Encoder saved at: {label_encoder_path}\n")

    # Print training label mapping (what was fitted on)
    print(f"Client {client_id} training label mapping: {dict(enumerate(label_encoder.classes_))}\n")

    # Get unique labels present in the test set
    test_unique_labels = set(test_data["category"])

    # Find which labels were in training vs unseen in test
    seen_labels = test_unique_labels.intersection(label_encoder.classes_)
    unseen_labels = test_unique_labels - seen_labels

    print(f"Client {client_id} test label mapping (seen): {seen_labels}\n")
    if unseen_labels:
        print(f"Client {client_id} test label mapping (unseen, assigned -1): {unseen_labels}\n")
    else:
        print(f"Client {client_id} test label mapping: No unseen labels, all categories were in training.\n")

    print(f"Train input shape: {X_train_processed.shape}, Train output shape: {y_train.shape}\n")
    print(f"Test input shape: {X_test_processed.shape}, Test output shape: {y_test.shape}\n")
    
    return X_train_processed, X_test_processed, y_train, y_test, tokenizer, label_encoder

# Create model with input parameters as number of rows and number of classes
def create_model(input_shape, num_classes):
    '''
    1) Create a simple sequential model with embedding layer, global average pooling, dense layer with relu activation and output layer with softmax activation:
        a) Embedding layer: This layer will take 25 length sequence of words and convert them into 64 dimensional vector(64 features representing the word) and the output shape will be (25,64) where 25 is the maximum length of the input sentence that we are giving to the model and the output will be a 25*64 matrix as every word of that sentence will be vectorized into 64 characteristics.The 800 is the vocabulary size which is the number of unique words in the dataset that the model needs to expect.
        b) Global Average Pooling: The input given to this layer will be the 25*64 matrix and the output will be a 64 dimensional vector of (64,) shape which is the average of the features(64 averaged features) which helps to tells you how important each feature is on average in the entire sequence i.e it summarizes the sentence.
        c) Dense layer: This layer has 64 neurons and each neuron receives all 64 feature values as inputs.Every neuron learns to detect or respond to a specific combination of those features.Each neuron applies : z = (w1 * x1) + (w2 * x2) + ... + (w64 * x64) + b where w1,w2,...,w64 are the weights and x1,x2,...,x64 are the input features and b is the bias.Then it is passed through ReLU activation function a non-linear function: f(z) = max(0,z) which helps the model to ignore irrelevant patterns (negative values are set to 0) and focuse on important patterns (positive values are kept as they are).
        d) Dense layer: This layer has 11 neurons which is the number of classes in the dataset in this case now it combines the 64 features into a single value (or logit) for each class.For neuron j in the output layer: z_j = (w1_j * x1) + (w2_j * x2) + ... + (w64_j * x64) + b_j where w1_j,w2_j,...,w64_j are the weights and x1,x2,...,x64 are the input features and b_j is the bias.Each neuron in this dense layer corresponds to one of the 11 classes and then the softmax activation takes the 11 logits and converts them into probabilities(11 probability outcomes i.e (11,)) that sum to 1.This way, the model gives a probability for each class, and the class with the highest probability is the prediction.
    '''

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(800, 64),
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

class BitextClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X_train, self.X_test, self.y_train, self.y_test, self.tokenizer, self.label_encoder = load_client_data(client_id)
        self.model = create_model(self.X_train.shape[1], len(np.unique(self.y_train)))
        self.model_path = os.path.join(CLIENT_MODEL_DIR, f'client{client_id}_model.keras')
        self.local_round = 0 # Counter for local rounds
        
    def get_parameters(self, config):
        #This function returns the weights of the model and config is used if we need layer specific weights.
        weights = self.model.get_weights()
        return weights

    def fit(self, parameters, config):
        
        # Increment the round counter
        self.local_round += 1

        print(f"\n######################## STEP 1 for Round {self.local_round}: Local Model Training for Client {self.client_id} ######################\n", flush=True)
        
        # 1) Set the model’s weights to the values passed as parameters.
        self.model.set_weights(parameters)
        print("\nModel Weights Before training:", flush=True)
        for i, weight in enumerate(self.model.get_weights()):
            print(f"\nWeight {i+1}: {weight.shape}\n{weight}", flush=True)

        # 2) Train the model for 5 epochs(rounds) with a batch size of 64 and a validation split of 20%.The training data is used to adjust the weights of the model and the validation data is used to evaluate the model after each epoch to see how well it is generalizing to unseen data.Batch size controls how many samples the model processes before updating weights. A smaller batch size means that the model is updated more often and the learning has more variance. A larger batch size means that the model is updated less often and the learning has less variance.Verbose is used for displaying the training process.
        print(f"\nStarting local training for client {self.client_id}...\n")
        history = self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=2)
        try:
            self.model.save(self.model_path)
            print(f"\nModel for client {self.client_id} saved successfully at {self.model_path}\n", flush=True)
        except Exception as e:
            print(f"\nError saving model for client {self.client_id}: {str(e)}", flush=True)
        
        # 3) Accessing the loss and accuracy of the model after training
        loss = history.history['loss'][-1]
        accuracy = history.history['accuracy'][-1]

        print("\nModel Weights After Training:",flush=True)
        for i, weight in enumerate(self.model.get_weights()):
            print(f"\nWeight {i+1}: {weight.shape}\n{weight}", flush=True)

        print("\n============= Local Training Completed =============",flush=True)
        print("============= Sending updated local model to server =============\n",flush=True)

        return self.model.get_weights(), len(self.X_train), {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        # 1) Set the model’s weights to the values passed as parameters.
        self.model.set_weights(parameters)

        print(f"\n################## STEP 2 for Round {self.local_round}: Local and Global Model Testing on Clients Side and updating the Client Model for client {self.client_id} ######################\n", flush=True)

        # 2) Evaluate the model on the clients local training data and return the loss and accuracy
        local_loss, local_accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        print(f"\nClient {self.client_id} - Local Evaluation Loss: {local_loss:.4f}, Accuracy: {local_accuracy:.4f}\n",flush=True)

        # 3) Evaluate the model on the global test data and return the loss and accuracy
        global_loss, global_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print(f"\nClient {self.client_id} - Global Evaluation Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}\n",flush=True)


        print(f"============= Client {self.client_id} Testing Completed =============",flush=True)
        print(f"============= Sending evaluation results to server ==================\n",flush=True)

        #The Evaluate Res object is returned which contains the loss, number of testing examples and the metric dictionary containing additional information.
        return local_loss, len(self.X_train), {"local_accuracy": local_accuracy,"global_loss": global_loss,"global_accuracy": global_accuracy,"X_test_len": len(self.X_test)}

# Start client (run this with client_id=0 to NUM_CLIENTS-1 in separate processes)
if __name__ == "__main__":
    client_id = int(sys.argv[1])
    print(f"\n*******************************Client {client_id} is running*****************************\n")

    # Convert NumPyClient to a Client instance
    client = BitextClient(client_id).to_client()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client
    )

    print(f"\n============= Client {client_id} Completed with updating with the final global model =============\n")