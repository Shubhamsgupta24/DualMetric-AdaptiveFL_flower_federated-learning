# Project Name
Colaborative Model Training in NLP using Federated Learning

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/Shubhamsgupta24/FlowerTensorflow.git

### 2. Create the Conda environment
conda env create -f environment.yml

### 3. Activate the environment
conda activate myenv

#### 3.1 Warning while Activating - The VS Code Editor might not identify the python interpreter so it is necessary to check or by selecting manually the python interpreter
conda info --envs
In VS Code select Python Interpreter

<img width="965" alt="image" src="https://github.com/user-attachments/assets/d04d3824-0933-4e21-913c-1bd580750141" />

### 4. Run Client and Server Python files in different terminals
python3 server.py
python3 client.py {client_id} , where {client_id} will be numerical values from 0 to NUM_CLIENTS in the code.
