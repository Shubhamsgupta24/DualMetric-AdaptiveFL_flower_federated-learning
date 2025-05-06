# Project Name
DualMetric-Adaptive FL: Our Fairness-Aware Federated Learning Framework for NLP

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/Shubhamsgupta24/FlowerTensorflow.git

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run Code
./run_federated.sh

## Code Instructions

### 1. Overview
This section provides a brief explanation of each key Python file in the project to help understand the code structure and logic for further modifications and contributions.

### 2. File Descriptions

#### 2.1 server.py
Handles the server-side logic for federated learning. It initializes the Flower server and sets up the Custom strategy by importing elements from FedAvg class.

In this the Global Variables NUM_CLIENTS and NUM_ROUNDS are to be taken care of.For caution try initially with fewer number of rounds( For eg. NUM_ROUNDS=10).

In server side code the server first initializes the code and then requests initial parameters from all the clients( refer FlowerDataFlow Diagram in project_materials). After that the server takes in the local model weights from all the clients and aggragtes the weights in aggregate_fit and returns the global updated model to all the clients. After the clients perform the evaluation of global model on the global test data and local training data and sends the evaulation results to the server. The server then aggregates the results in aggragate_evaluate as well as evaluates the results to set up the proximal regularization paramter dynamically according to intra-client and inter-client anaysis( refer report from project_materials) in tune_client_hyperparameters. Finally the server configures the final regularization parameter(mu) for every client and sends the updated mu via Configure_Fit Packet.

#### 2.2 client.py
Contains the client-side code. Each client loads local data, trains a model, and communicates updates to the server.

In the client side code first initialization is done in init where we create a local model architecture and also load data, global tokenizer and label encoder.
After that we start the local training of the data in fit that is native to that client and send the model to the server to perform aggregation in a federated learning setup.
The server sends the global updated model to the client and then the client performs evaluation of two types for local accuracy i.e the accuracy of global model on local data that the client is being trained on and the global accuracy i.e the accuracy of global model on unknown test data.The global accuracy would be same for all the number of clients.
After the server performs analysis the client again starts with local training but with changed regularization parameter extraced from Config_Fit packet i.e used in fed_prox loss function in fit.

#### 2.3 data_prep.py
Responsible for preprocessing text data, tokenizing inputs, and preparing label encodings for training and testing.

Check for the global variables and especially the Preprocessing Flag.If you are running the code for the first time set the flag to False else after generating the corrected_dataset.csv you can set it to True as it saves time and resources.

It firsts loads the data from Dataset and then preprocesses it by correcting the spelling and generating a corrected_dataset in Dataset.
After that it creates a global tokenizer and label_encoder from the entire dataset so that we get a dictionary with all the words in dataset for common word embeddings across the clients.
Then it splits the data into the training data and testing data.The TEST_SIZE defines the partition ratio of training and testing data.
After we get the training data accoridng to the partition type and number of clients we divide the data accoridngly.You can try with different data partitions that are commented out.

The data partitions were hard coded accoriding to different datasets that were tries during the experimentations so be careful while creating data partitions.

- You can run "python3 data_prep.py" to try with the partitions first before proceeding with the federated learning setup.

#### 2.4 visualisations.py
These codefile is being imported in other codefiles which deines the visulation code functions for different parts of the code.

It visualizes the data partitions, global accuracy across rounds, local accuracy across rounds and the tradeoff between accuracy and variance in the setup.Moreover, it also stores the histories of local and global accuracy for the clients for the rounds.

#### 2.5 predictions.py
It uses streamlit library for making a GUI wherein the user can input query and the final global model after NUM_ROUNDS would give probability of the predicted classes.It takes in input as the saved model in the model directory.

These helps to debug the code and make improvements in the existing code.

These file is optional and may be of use to the coder.

#### 2.6 run_federated.sh
This is a shell script to automate the running of Federated Learning setup although some manual checks are required to ensure consistency and proper flow.

It firsts removes and recreates the directory for a new environment each time we run the code.
Then it runs data_prep.py for data preparation.
After the data is prepared the server and client codes are run.The server being first for listening.

An alternative to manually learn the process would be that istead of running this automated script the user can follow these steps:
 
- Run "python3 data_prep.py"
- Run "python3 server.py"
- Run "python3 client.py 0" , "python3 client.py 1" , python3 client.py 2" and so on till (NUM_CLIENTS - 1) , all in different terminals.

All the log files generated are stored in OUTPUTS Directory.

## Novelty
Proposed DualMetric-Adaptive FL improves clients fariness in the federated learning setup.

To check for the improvement run these code and then comment out the self.tune_client_hyperparameters(server_round) in aggregate_evaluate function in server.py which will disbale the dynamic changes made to regularization paramter so the default regularization paramter value(mu) will be taken by client and not the one guided by the server.