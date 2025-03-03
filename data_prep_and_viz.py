import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Global variables
TRAIN_DIR = './Dataset/Train'
TEST_DIR = './Dataset/Test'
VISUAL_DIR = './Dataset/Visualizations'
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

def visualize_client_datasets():
    """
    Visualizes the category distribution for all client datasets stored in the Train directory.
    Ensures all categories (from all clients) appear in every client plot.
    """

    client_files = []  # List to store client dataset filenames

    # Find all client dataset files
    for file in os.listdir(TRAIN_DIR):
        if file.startswith('train_data_client') and file.endswith('.csv'):
            client_files.append(file)

    if not client_files:
        print("No client datasets found! Ensure the data is prepared first.")
        return

    # Gather all unique categories from client datasets
    all_categories = set()
    client_datasets = {}

    for client_file in client_files:
        client_data = pd.read_csv(os.path.join(TRAIN_DIR, client_file))
        unique_categories = set(client_data['category'].unique())
        all_categories.update(unique_categories)  # Collect all unique categories
        client_datasets[client_file] = client_data  # Store data for reuse

    # Convert to sorted list for consistency
    all_categories = sorted(all_categories)

    # Define grid size for subplots
    num_clients = len(client_files)
    cols = 4  # 4 clients per row
    rows = (num_clients // cols) + (num_clients % cols > 0)  # Calculate needed rows

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, client_file in enumerate(client_files):
        client_data = client_datasets[client_file]

        # Count category distribution
        category_counts = client_data['category'].value_counts()

        # Ensure all categories exist (even if 0)
        category_counts = category_counts.reindex(all_categories, fill_value=0)

        # Plot each client’s distribution
        ax = axes[i]
        sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, 
                    dodge=False, legend=False, palette="viridis", ax=ax)

        # Adjust y-axis limit for text visibility
        max_count = category_counts.max()
        ax.set_ylim(0, max_count * 1.25)  # Add 25% extra space on top

        # Display count on top of each bar
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", 
                        (p.get_x() + p.get_width() / 2, p.get_height() * 1.05),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

        ax.set_title(f"Client {i} Category Distribution")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        
        ax.set_xticks(range(len(all_categories)))  
        ax.set_xticklabels(all_categories, rotation=45, ha='right')

    # Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(top=0.9, hspace=0.5)

    # Save the figure
    graph_path = os.path.join(VISUAL_DIR, "client_category_distribution.png")
    try:
        plt.savefig(graph_path, dpi=300, bbox_inches="tight")  # Save without clipping
        print(f"Visualization saved successfully at: {graph_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

def visualize_test_dataset():
    """
    Visualizes the category distribution for the global test dataset.
    Saves the visualization as a PNG file.
    """

    # Load test dataset
    test_data = pd.read_csv(os.path.join(TEST_DIR, 'global_test_set.csv'))

    # Count category distribution
    category_counts = test_data['category'].value_counts()

    # Create a figure for visualization
    plt.figure(figsize=(20, 10))

    # Plot the test dataset distribution
    ax = sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, legend=False, palette="viridis")

    # Display count on top of each bar
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2, p.get_height()),  
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    plt.title("Global Test Dataset Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Save the figure
    graph_path = os.path.join(VISUAL_DIR, "test_category_distribution.png")
    try:
        plt.savefig(graph_path, dpi=300)
        print(f"Visualization saved successfully at: {graph_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

if __name__ == "__main__":
    data = load_data()
    train_data, test_data = split_data(data)
    save_test_data(test_data)

    # prepare_train_data_iid(train_data,8) # NUM_CLIENTS = 8
    prepare_train_data_noniid_0(train_data,11) # NUM_CLIENTS = 11
    # prepare_train_data_noniid_1(train_data) # NUM_CLIENTS = 6 by default

    visualize_client_datasets()
    visualize_test_dataset()