import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns

NUM_CLIENTS = 11
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
    """ Ensuring test directory exists and saving test dataset """
    os.makedirs(TEST_DIR, exist_ok=True)
    test_file = os.path.join(TEST_DIR, 'global_test_set.csv')
    try:
        test_data.to_csv(test_file, index=False)
        print(f"Global test set saved successfully at: {test_file}")
    except Exception as e:
        print(f"Error saving global test set: {e}")

def prepare_train_data_iid(train_data):
    """
    Prepare client-specific datasets by splitting the dataset equally among NUM_CLIENTS.
    This represents IID (Independent and Identically Distributed) data in Federated Learning.
    """
    # Ensure train directory exists
    os.makedirs(TRAIN_DIR, exist_ok=True)

    # Partitioning the dataset into NUM_CLIENTS equal splits
    data_splits = np.array_split(train_data, NUM_CLIENTS)

    # Save each client's data
    for client_id, client_data in enumerate(data_splits):
        client_file = os.path.join(TRAIN_DIR, f'train_data_client{client_id}.csv')
        try:
            client_data.to_csv(client_file, index=False)
            print(f"Client {client_id} IID dataset saved successfully at: {client_file}")
        except Exception as e:
            print(f"Error saving client {client_id} dataset: {e}")

def prepare_train_data_noniid_0(train_data):
    """
    Prepare client-specific datasets where each client is assigned a primary category
    and gets one sample from every other category.
    This is an example of Non-IID data distribution wherein we have considered case 1 of extreme biasness.
    """

    # Group train data by category
    grouped_data = train_data.groupby("category")
    categories = list(grouped_data.groups.keys())

    # Ensure NUM_CLIENTS does not exceed available categories
    if NUM_CLIENTS > len(categories):
        raise ValueError(f"NUM_CLIENTS ({NUM_CLIENTS}) is greater than available categories ({len(categories)}).")

    # Ensure train directory exists
    os.makedirs(TRAIN_DIR, exist_ok=True)

    # Assign categories to clients and create client-specific datasets
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

def visualize_client_datasets():
    """
    Visualizes the category distribution for all client datasets stored in the Train directory.
    Saves the visualization as a PNG file.
    """

    client_files = []  # List to store client dataset filenames

    # Find all client dataset files
    for file in os.listdir(TRAIN_DIR):
        if file.startswith('train_data_client') and file.endswith('.csv'):
            client_files.append(file)

    if not client_files:
        print("No client datasets found! Ensure the data is prepared first.")
        return

    # Define grid size for subplots
    num_clients = len(client_files)
    cols = 4  # 4 clients per row
    rows = (num_clients // cols) + (num_clients % cols > 0)  # Calculate needed rows

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, client_file in enumerate(client_files):
        # Load client dataset
        client_data = pd.read_csv(os.path.join(TRAIN_DIR, client_file))

        # Count category distribution
        category_counts = client_data['category'].value_counts()

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
        
        ax.set_xticks(range(len(category_counts.index)))  
        ax.set_xticklabels(category_counts.index, rotation=45, ha='right')

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

    # prepare_train_data_iid(train_data)
    prepare_train_data_noniid_0(train_data)

    visualize_client_datasets()
    visualize_test_dataset()