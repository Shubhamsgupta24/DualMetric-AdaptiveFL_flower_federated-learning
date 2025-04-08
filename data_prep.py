import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore 
from visualisations import visualize_client_datasets, visualize_test_dataset
from wordfreq import word_frequency, top_n_list
from symspellpy import SymSpell, Verbosity

# Global variables
TRAIN_DIR = './Dataset/Train'
TEST_DIR = './Dataset/Test'
VISUAL_DIR = './Visualizations'
# DATA_PATH='./Dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
DATA_PATH="./Dataset/Retail_bitext_dataset.csv"
TOKENIZER_PATH = "./models/Global/tokenizer.json"
LABEL_ENCODER_PATH = "./models/Global/label_encoder.pkl"
TEST_SIZE = 0.2 # Percentage of data for testing from the main dataset
PREPROCESSING_FLAG = True # Set to True if preprocessing is done else otherwise

def generate_dictionary(sym_spell):
    """Generate and print dictionary dynamically from wordfreq."""
    print("Generating SymSpell dictionary...\n")

    # Fetch the top 100K words and their frequencies
    dictionary_entries = []
    for word in top_n_list("en", 100000):
        freq = word_frequency(word, "en")
        scaled_freq = int(freq * 1e6)  # Scale frequency for SymSpell
        
        if scaled_freq > 0:
            # Add the word and its frequency to the SymSpell dictionary
            sym_spell.create_dictionary_entry(word, scaled_freq)
            dictionary_entries.append((word, scaled_freq))

    # Print the first 10 entries for debugging
    print("First 10 words in the dictionary:")
    for i, (word, freq) in enumerate(dictionary_entries[:10]):
        print(f"{i+1}. {word}: {freq}")

def correct_spelling(text,sym_spell):
    """Performs word segmentation and correct spelling of words in the text using SymSpell."""

    words = text.split()
    corrected_words = []
    
    for word in words:
        # Allows corrections for words with up to 3 character edits (insertions, deletions, replacements, or transpositions).
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=5)
        if suggestions:
            corrected_words.append(suggestions[0].term)  # Use the best suggestion
        else:
            corrected_words.append(word)  # Keep original word if no suggestion found
    
    corrected_text = " ".join(corrected_words)  # Join corrected words into a sentence

    return corrected_text

def load_data(sym_spell):
    """ Load and preprocess the dataset """
    # Load main dataset
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    # Clean and correct data
    data['instruction'] = (
        data['instruction']
        .astype(str)  # Ensure it's a string
        .str.lower()  # Convert to lowercase
        .str.replace(r"[^\w\s]|\d+", "", regex=True)  # Remove special characters and digits
        .apply(lambda text: correct_spelling(text, sym_spell))  # Correct spelling
        )

    # Save corrected dataset
    corrected_path = "./Dataset/Corrected_dataset.csv"
    data.to_csv(corrected_path, index=False)
    print(f"\nCorrected dataset saved to {corrected_path}\n")

    return data

def split_data(data, test_size=TEST_SIZE):
    """ Perform train-test split """
    return train_test_split(data, test_size=test_size, random_state=42)

def save_test_data(test_data):
    test_file = os.path.join(TEST_DIR, 'global_test_set.csv')
    try:
        test_data.to_csv(test_file, index=False)
        print(f"Global test set saved successfully at: {test_file}\n")
    except Exception as e:
        print(f"Error saving global test set: {e}")

def prepare_and_save_tokenizer_label_encoder(data,text_column,label_column):
    """Creates and saves a global tokenizer and label encoder using preprocessed Bittext.csv."""

    texts = data[text_column].astype(str).tolist()  # Ensure text data is in string format
    labels = data[label_column].tolist()
    
    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    
    # Save tokenizer
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_json(), f, ensure_ascii=False)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    # Save label encoder
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"Global label mapping: {dict(enumerate(label_encoder.classes_))}\n")
    print(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
    print("Tokenizer and Label Encoder saved successfully!\n")

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

def prepare_train_data_noniid_2(train_data):
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

        # Get categories NOT assigned to this client
        all_categories = set(grouped_data.keys())
        assigned_categories = set(categories["primary"] + categories["secondary"])
        unknown_categories = list(all_categories - assigned_categories)

        # Sample n rows from each unknown category
        extra_data = pd.concat([grouped_data[cat].sample(n=2, random_state=42, replace=True) for cat in unknown_categories])
        client_data = pd.concat([primary_data, secondary_data, extra_data]).sample(frac=1.0, random_state=42)  # Shuffle

        # Convert category indexes back to labels
        client_data["category"] = label_encoder.inverse_transform(client_data["category"])

        # Save to CSV
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} non-IID dataset saved successfully at: {client_file}")
        except Exception as e:
            print(f"Error saving client {client_id} dataset: {e}")

def prepare_train_data_noniid_3(train_data):
    '''
    Prepare client-specific datasets based on intents using non-IID partitioning (case 2 style).
    Assumes there are 27 unique intents.
    '''

    # 1) Encode intent labels as numeric indexes (0-26)
    label_encoder = LabelEncoder()
    train_data['intent'] = label_encoder.fit_transform(train_data['intent'])

    # 2) Regroup dataset by intent index
    grouped_data = {idx: group for idx, group in train_data.groupby("intent")}

    # 3) Define client-specific primary/secondary intent allocations
    client_intents = {
        0: {"primary": [0, 6, 12], "secondary": [1, 7]},
        1: {"primary": [1, 7, 13], "secondary": [2, 8]},
        2: {"primary": [2, 8, 14], "secondary": [3, 9]},
        3: {"primary": [3, 9, 15], "secondary": [4, 10]},
        4: {"primary": [4, 10, 16], "secondary": [5, 11]},
        5: {"primary": [5, 11, 17], "secondary": [6, 12]},
        6: {"primary": [6, 12, 18], "secondary": [7, 13]},
        7: {"primary": [7, 13, 19], "secondary": [8, 14]},
        8: {"primary": [8, 14, 20], "secondary": [9, 15]},
        9: {"primary": [9, 15, 21], "secondary": [10, 16]},
        10: {"primary": [10, 16, 22], "secondary": [11, 17]},
        11: {"primary": [11, 17, 23], "secondary": [0, 18]}
    }

    # 4) Assign data to each client
    for client_id, intents in client_intents.items():
        primary_data = pd.concat([
            grouped_data[intent].sample(frac=0.5, random_state=42)
            for intent in intents["primary"]
        ])

        secondary_data = pd.concat([
            grouped_data[intent].sample(frac=0.25, random_state=42)
            for intent in intents["secondary"]
        ])

        all_intents = set(grouped_data.keys())
        assigned_intents = set(intents["primary"] + intents["secondary"])
        unknown_intents = list(all_intents - assigned_intents)

        extra_data = pd.concat([
            grouped_data[intent].sample(n=2, random_state=42, replace=True)
            for intent in unknown_intents
        ])

        client_data = pd.concat([primary_data, secondary_data, extra_data]).sample(frac=1.0, random_state=42)

        # Convert back to original intent labels
        client_data["intent"] = label_encoder.inverse_transform(client_data["intent"])

        # Save to CSV
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} non-IID intent dataset saved at: {client_file}")
        except Exception as e:
            print(f"Error saving intent data for client {client_id}: {e}")

def prepare_train_data_noniid_4(train_data):
    '''
    Automatically generate client-specific datasets based on non-IID splitting.
    Works for any number of intents (e.g., 46).
    '''
    intents_per_client=4
    secondary_per_client=2
    extra_samples_per_intent=2

    # Encode labels
    label_encoder = LabelEncoder()
    train_data['intent'] = label_encoder.fit_transform(train_data['intent'])

    # Group data by intent
    grouped_data = {idx: group for idx, group in train_data.groupby("intent")}
    total_intents = len(grouped_data)

    # Calculate number of clients needed
    num_clients = total_intents // intents_per_client

    client_intents = {}
    for i in range(num_clients):
        start = i * intents_per_client
        primary = list(range(start, start + intents_per_client))
        secondary = [(x + 1) % total_intents for x in primary[:secondary_per_client]]
        client_intents[i] = {"primary": primary, "secondary": secondary}

    # Handle remaining intents (if not divisible exactly)
    remaining_intents = set(range(total_intents)) - set(
        sum([v["primary"] for v in client_intents.values()], [])
    )
    if remaining_intents:
        for idx, intent in enumerate(remaining_intents):
            client_id = idx % num_clients
            client_intents[client_id]["primary"].append(intent)

    # Generate client datasets
    for client_id, intents in client_intents.items():
        primary_data = pd.concat([
            grouped_data[i].sample(frac=0.9, random_state=42)
            for i in intents["primary"]
        ])

        secondary_data = pd.concat([
            grouped_data[i].sample(frac=0.3, random_state=42)
            for i in intents["secondary"]
        ])

        # Determine unknown intents
        assigned = set(intents["primary"] + intents["secondary"])
        unknown = list(set(grouped_data.keys()) - assigned)

        extra_data = pd.concat([
            grouped_data[i].sample(n=extra_samples_per_intent, replace=True, random_state=42)
            for i in unknown
        ])

        client_data = pd.concat([primary_data, secondary_data, extra_data]).sample(frac=1.0, random_state=42)

        # Convert back to original intent labels
        client_data['intent'] = label_encoder.inverse_transform(client_data['intent'])

        # Save CSV
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} data saved at: {client_file}")
        except Exception as e:
            print(f"Error saving data for client {client_id}: {e}")


if __name__ == "__main__":

    preprocessing_done = PREPROCESSING_FLAG

    if not preprocessing_done:
        # Preprocessing not done and thereby creating a corrected dataset
        print("Preprocessing not done. Initializing Preprocessing...\n\n")
        # Initialize SymSpell for spelling correction
        sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=10)

        # Generate dictionary for SymSpell
        generate_dictionary(sym_spell)

        # Data preprocessing and splitting
        data = load_data(sym_spell)
    else:
        # Preprocessing done and thereby loading the corrected dataset
        print("Preprocessing done. Loading corrected dataset...\n\n")
        data = pd.read_csv("./Dataset/Corrected_dataset.csv")
    
    train_data, test_data = split_data(data)
    save_test_data(test_data)
    prepare_and_save_tokenizer_label_encoder(data, 'instruction', 'intent')

    # prepare_train_data_iid(train_data,6) # NUM_CLIENTS = 6
    # prepare_train_data_noniid_0(train_data,11) # NUM_CLIENTS = 11
    # prepare_train_data_noniid_1(train_data) # NUM_CLIENTS = 6 by default
    # prepare_train_data_noniid_2(train_data) # NUM_CLIENTS = 6 (n rows each)
    # prepare_train_data_noniid_3(train_data) # NUM_CLIENTS = 11 (n rows each)
    prepare_train_data_noniid_4(train_data)

    visualize_client_datasets(TRAIN_DIR, VISUAL_DIR, "intent")
    visualize_test_dataset(TEST_DIR, VISUAL_DIR, "intent")
