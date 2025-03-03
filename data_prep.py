import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
from visualisations import visualize_client_datasets, visualize_test_dataset

# Global variables
TRAIN_DIR = './Dataset/Train'
TEST_DIR = './Dataset/Test'
VISUAL_DIR = './Visualizations'
TEST_SIZE = 0.1 # Percentage of data for testing from the main dataset

def load_data():
    # Load main dataset
    data_path = './Dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    # Clean data
    data['instruction'] = data['instruction'].str.replace(r"[^\w\s]", "", regex=True)

    return data

def split_data(data, test_size=TEST_SIZE):
    """ Perform train-test split """
    return train_test_split(data, test_size=test_size, random_state=42)

def save_test_data(test_data):
    test_file = os.path.join(TEST_DIR, 'global_test_set.csv')
    try:
        test_data.to_csv(test_file, index=False)
        print(f"Global test set saved successfully at: {test_file}")
    except Exception as e:
        print(f"Error saving global test set: {e}")

def prepare_train_data_iid(train_data,NUM_CLIENTS):
    """
    Prepare client-specific datasets by splitting the dataset equally among NUM_CLIENTS.
    This represents IID (Independent and Identically Distributed) data in Federated Learning.
    """

    # 1) Partitioning the dataset into NUM_CLIENTS equal splits
    data_splits = np.array_split(train_data, NUM_CLIENTS)

    # 2) Save each client's data
    for client_id, client_data in enumerate(data_splits):
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} IID dataset saved successfully at: {client_file}")
        except Exception as e:
            print(f"Error saving client {client_id} dataset: {e}")

def prepare_train_data_noniid_0(train_data,NUM_CLIENTS):
    """
    Prepare client-specific datasets where each client is assigned a primary category
    and gets one sample from every other category.
    This is an example of Non-IID data distribution wherein we have considered case 1 of extreme biasness.
    """

    # 1) Group train data by category
    grouped_data = train_data.groupby("category")
    categories = list(grouped_data.groups.keys())

    # 2) Ensure NUM_CLIENTS does not exceed available categories
    if NUM_CLIENTS > len(categories):
        raise ValueError(f"NUM_CLIENTS ({NUM_CLIENTS}) is greater than available categories ({len(categories)}).")

    # 3) Assign categories to clients and create client-specific datasets
    for client_id, category in enumerate(categories[:NUM_CLIENTS]):  # Assign each client a primary category
        client_data = grouped_data.get_group(category)

        # Add one sample from each other category
        for other_category in categories:
            if other_category != category:
                try:
                    sample_row = grouped_data.get_group(other_category).sample(n=1, random_state=42, replace=True)
                    client_data = pd.concat([client_data, sample_row])
                except Exception as e:
                    print(f"Error adding sample from category {other_category} to Client {client_id}: {e}")

        # Define client CSV path
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')

        # Save client-specific dataset with error handling
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} Non-IID dataset saved successfully at: {client_file}")
        except Exception as e:
            print(f"Error saving client {client_id} dataset: {e}")

def prepare_train_data_noniid_1(train_data):
    '''
    Prepare client-specific datasets where each client is assigned a primary category
    and gets a random sample from a subset of other categories.
    This is an example of Non-IID data distribution wherein we have considered case 2 of biasness and we have hardcoded the distribution taking consideration of 6 Clients.
    '''

    # 1) Encode category labels as numeric indexes (0-10)
    label_encoder = LabelEncoder()
    train_data['category'] = label_encoder.fit_transform(train_data['category'])  # Convert to numeric indexes

    # 2) Regroup dataset using numeric indexes instead of category names
    grouped_data = {idx: group for idx, group in train_data.groupby("category")}  # Store groups in a dictionary

    # 3) Define client-specific category allocation (no row sharing)
    client_categories = {
        0: {"primary": [0, 6], "secondary": [10, 5, 4]},
        1: {"primary": [1, 6], "secondary": [9, 0, 5]},
        2: {"primary": [2, 7], "secondary": [9, 1, 0]},
        3: {"primary": [3, 7], "secondary": [9, 2, 1]},
        4: {"primary": [4, 8], "secondary": [10, 3, 2]},
        5: {"primary": [5, 8], "secondary": [10, 4, 3]}
    }

    # 4) Assign data to clients (ensuring no row is shared)
    for client_id, categories in client_categories.items():
        primary_data = pd.concat([grouped_data[cat].sample(frac=0.50, random_state=42) for cat in categories["primary"]])
        secondary_data = pd.concat([grouped_data[cat].sample(frac=0.33 if cat == 10 else 0.25, random_state=42) for cat in categories["secondary"]])
        client_data = pd.concat([primary_data, secondary_data]).sample(frac=1.0, random_state=42)  # Shuffle data

        # Convert category indexes back to labels
        client_data["category"] = label_encoder.inverse_transform(client_data["category"])

        # Save to CSV
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} non-IID dataset saved successfully at: {client_file}")
        except Exception as e:
            print(f"Error saving client {client_id} dataset: {e}")

if __name__ == "__main__":
    data = load_data()
    train_data, test_data = split_data(data)
    save_test_data(test_data)

    # prepare_train_data_iid(train_data,8) # NUM_CLIENTS = 8
    prepare_train_data_noniid_0(train_data,11) # NUM_CLIENTS = 11
    # prepare_train_data_noniid_1(train_data) # NUM_CLIENTS = 6 by default

    visualize_client_datasets(TRAIN_DIR, VISUAL_DIR)
    visualize_test_dataset(TEST_DIR, VISUAL_DIR)