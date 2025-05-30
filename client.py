import tensorflow as tf
import flwr as fl
import pandas as pd
import numpy as np
import os, sys
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

'''
CAUTION:
1) Ensure that the dataset is present in the Dataset folder.
'''

# Global variables
MODEL_DIR="./models"
GLOBAL_TOKENIZER_PATH = "./models/Global/tokenizer.json"
GLOBAL_LABEL_ENCODER_PATH = "./models/Global/label_encoder.pkl"

# Setting print options for better readability where precision is the decimal digits to be printed and threshold is the number of array elements which triggers summarization in a numpy array.
np.set_printoptions(precision=5, threshold=50)

# Client-specific data loading
def load_client_data(client_id, text_column, label_column):
    # 1) Get train and test data for the current client
    train_data = pd.read_csv(f'./Dataset/Train/train_data_client{client_id}.csv')
    test_data = pd.read_csv('./Dataset/Test/global_test_set.csv')
    
    # 2) Load global tokenizer
    with open(GLOBAL_TOKENIZER_PATH, "r") as f:
        tokenizer = tokenizer_from_json(json.load(f))

    X_train = train_data[text_column].values
    X_test = test_data[text_column].values

    X_train_processed = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=15)
    X_test_processed = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=15)
    
    # 3) Loading global label encoder
    with open(GLOBAL_LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    y_train = label_encoder.transform(train_data[label_column].values)
    
    # 4) Encode test labels, assign -1 to unseen categories
    y_test = []
    for category in test_data[label_column]:
        if category in label_encoder.classes_:
            y_test.append(label_encoder.transform([category])[0])
        else:
            y_test.append(-1)  # Assign -1 for unseen categories
    y_test = np.array(y_test)
    
    # 5) Saving tokenizer and label encoder for future use
    tokenizer_path = os.path.join(MODEL_DIR, f"Client{client_id}/tokenizer.json")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.to_json(), f)
    print(f"Tokenizer saved at: {tokenizer_path}\n")

    label_encoder_path = os.path.join(MODEL_DIR, f"Client{client_id}/label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label Encoder saved at: {label_encoder_path}\n")

    # Print training label mapping (what was fitted on)
    print(f"Client {client_id} training label mapping: {dict(enumerate(label_encoder.classes_))}\n")

    # Get unique labels present in the test set
    test_unique_labels = set(test_data[label_column])

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
def create_model(input_shape, num_classes, tokenizer):
    '''
    1) Create a simple sequential model with embedding layer, global average pooling, dense layer with relu activation and output layer with softmax activation:
        a) Embedding layer: This layer will take 15 length sequence of words and convert them into 64 dimensional vector(64 features representing the word) and the output shape will be (25,64) where 25 is the maximum length of the input sentence that we are giving to the model and the output will be a 25*64 matrix as every word of that sentence will be vectorized into 64 characteristics.The 800 is the vocabulary size which is the number of unique words in the dataset that the model needs to expect.
        b) Global Average Pooling: The input given to this layer will be the 25*64 matrix and the output will be a 64 dimensional vector of (64,) shape which is the average of the features(64 averaged features) which helps to tells you how important each feature is on average in the entire sequence i.e it summarizes the sentence.
        c) Dense layer: This layer has 64 neurons and each neuron receives all 64 feature values as inputs.Every neuron learns to detect or respond to a specific combination of those features.Each neuron applies : z = (w1 * x1) + (w2 * x2) + ... + (w64 * x64) + b where w1,w2,...,w64 are the weights and x1,x2,...,x64 are the input features and b is the bias.Then it is passed through ReLU activation function a non-linear function: f(z) = max(0,z) which helps the model to ignore irrelevant patterns (negative values are set to 0) and focuse on important patterns (positive values are kept as they are).
        d) Batch Normalization: This layer normalizes the activations of the previous layer at each batch i.e it scales and shifts the activations to have zero mean and unit variance which helps in faster training and higher overall accuracy.
        e) Dense layer: This layer has 11 neurons which is the number of classes in the dataset in this case now it combines the 64 features into a single value (or logit) for each class.For neuron j in the output layer: z_j = (w1_j * x1) + (w2_j * x2) + ... + (w64_j * x64) + b_j where w1_j,w2_j,...,w64_j are the weights and x1,x2,...,x64 are the input features and b_j is the bias.Each neuron in this dense layer corresponds to one of the 11 classes and then the softmax activation takes the 11 logits and converts them into probabilities(11 probability outcomes i.e (11,)) that sum to 1.This way, the model gives a probability for each class, and the class with the highest probability is the prediction.
    '''

    vocab_size = len(tokenizer.word_index) + 1

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dense(num_classes, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.0001))
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
        self.X_train, self.X_test, self.y_train, self.y_test, self.tokenizer, self.label_encoder = load_client_data(client_id, "instruction", "intent")
        self.model = create_model(self.X_train.shape[1], len(np.unique(self.y_train)), self.tokenizer)
        self.model_path = os.path.join(MODEL_DIR, f'Client{client_id}/model.keras')
        self.local_round = 0 # Counter for local rounds
        
    def get_parameters(self, config):
        #This function returns the weights of the model and config is used if we need layer specific weights.
        weights = self.model.get_weights()
        return weights

    def fit(self, parameters, config):
        
        # Increment the round counter
        self.local_round += 1

        print(f"\n######################## STEP 1 for Round {self.local_round}: Local Model Training for Client {self.client_id} ######################\n", flush=True)
        
        # 1) Set the model’s weights to the values passed as parameters and storing global weights for fedprox loss.
        self.model.set_weights(parameters)
        global_weights = parameters  # Store global weights for FedProx

        # 2) Get the trainable weights correctly by matching layer names
        global_trainable_weights = []
        trainable_var_names = {v.name for v in self.model.trainable_variables}

        print ("Trainable Variable Names:",flush=True)
        for i, var in enumerate(self.model.trainable_variables):
            print(f"{i}: {var.name}",flush=True)

        print("Non-Trainable Variable Names:",flush=True)
        for i, var in enumerate(self.model.non_trainable_variables):
            print(f"{i}: {var.name}",flush=True)

        for w, var in zip(global_weights, self.model.weights):  
            if var.name in trainable_var_names:  
                global_trainable_weights.append(w)

        print("\n=== Local Model Weights ===", flush=True)
        for i, weight in enumerate(self.model.trainable_weights):
            print(f"Layer {i}: Shape {weight.shape}", flush=True)

        print("\n=== Global Weights Received from Server (have excluded non-trainable weights) ===", flush=True)
        for i, weight in enumerate(global_trainable_weights):
            print(f"Layer {i}: Shape {weight.shape}", flush=True)

        print("\nModel Weights Before training :", flush=True)
        for i, weight in enumerate(self.model.get_weights()):
            print(f"\nWeight {i+1}: {weight.shape}\n{weight}", flush=True)

        print(f"\nStarting local training for client {self.client_id}...\n")
        
        # 3) Extracting FedProx mu (regularization parameter) - mu controls how much clients deviate from the global model. Higher mu makes local updates more similar to the global model and lower mu makes local updates more similar to the local data.
        mu = config.get("proximal_mu", 0.01)

        # 4) Extract the lr_factor from the config which is used to adjust the learning rate of the model. The learning rate is multiplied by this factor to adjust the learning rate of the model.
        lr_factor = config.get("lr_factor", 1.0)  # Server-sent lr_factor, default 1.0
        base_lr = 0.001  # Base learning rate

        print(f"Client {self.client_id} - Using mu: {mu:.4f}, lr_factor: {lr_factor:.4f}, effective learning rate: {base_lr * lr_factor:.6f}", flush=True)

        # 5) Defining a custom loss function to include the proximal term used in FedProx strategy
        def fedprox_loss(y_true, y_pred):
            base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
            # Compute proximal regularization term
            prox_term = 0
            for w, w_g in zip(self.model.trainable_weights, global_trainable_weights):
                squared_difference = tf.square(w - w_g)  # Element-wise square
                sum_squared = tf.reduce_sum(squared_difference)  # Sum of squared differences
                prox_term += sum_squared  # Accumulate across all layers

            # Final loss = base loss + FedProx regularization term
            final_loss = base_loss + (mu / 2) * prox_term
            return final_loss

        # 6) Compile the model again with FedProx loss which includes the proximal term and dynamic learning rate.
        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr * lr_factor)  # Apply lr_factor to base_lr
        self.model.compile(optimizer=optimizer, loss=fedprox_loss, metrics=['accuracy'])

        # 7) Early stopping callback to stop training when the loss stops decreasing and it will continue for 3 epochs before stopping and restore_best_weights is used to restore the weights of the model when the training stops.
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # 8) Train the model for 5 epochs(rounds) with a batch size of 64 and a validation split of 30%.The training data is used to adjust the weights of the model and the validation data is used to evaluate the model after each epoch to see how well it is generalizing to unseen data.Batch size controls how many samples the model processes before updating weights. A smaller batch size means that the model is updated more often and the learning has more variance. A larger batch size means that the model is updated less often and the learning has less variance.Verbose is used for displaying the training process.
        history = self.model.fit(self.X_train, self.y_train, epochs=5, validation_split=0.4, verbose=2,callbacks=[early_stopping])
        
        # 9) Accessing the loss and accuracy of the model after training
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

        # 2) Save the model to the specified path
        try:
            self.model.save(self.model_path)
            print(f"\nModel for client {self.client_id} saved successfully at {self.model_path}\n", flush=True)
        except Exception as e:
            print(f"\nError saving model for client {self.client_id}: {str(e)}", flush=True)

        print(f"\n################## STEP 2 for Round {self.local_round}: Local and Global Model Testing on Clients Side and updating the Client Model for client {self.client_id} ######################\n", flush=True)

        # 3) Evaluate the model on the clients local training data and return the loss and accuracy
        local_loss, local_accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=2)
        print(f"\nClient {self.client_id} - Local Evaluation Loss: {local_loss:.4f}, Accuracy: {local_accuracy:.4f}\n",flush=True)

        # 4) Evaluate the model on the global test data and return the loss and accuracy
        global_loss, global_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print(f"\nClient {self.client_id} - Global Evaluation Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}\n",flush=True)

        print(f"============= Client {self.client_id} Testing Completed =============",flush=True)
        print(f"============= Sending evaluation results to server ==================\n",flush=True)

        #The Evaluate Res object is returned which contains the loss, number of testing examples and the metric dictionary containing additional information.
        return local_loss, len(self.X_train), {"local_accuracy": local_accuracy,"global_loss": global_loss,"global_accuracy": global_accuracy,"X_test_len": len(self.X_test)}


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