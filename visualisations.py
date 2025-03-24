import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_client_datasets(TRAIN_DIR, VISUAL_DIR):
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
    cols = 3  # 4 clients per row
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

def visualize_test_dataset(TEST_DIR, VISUAL_DIR):
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

def visualize_global_accuracy_clients(global_accuracy_hist,VISUAL_DIR):
    """
    Plots global accuracy for each client and saves the figure as a PNG.
    
    Parameters:
        global_accuracy_hist (dict): Dictionary with client IDs as keys and accuracy lists as values.
        VISUAL_DIR: Directory where the PNG file will be saved.
    """
    plt.figure(figsize=(10, 5))
    for idx, (client, acc) in enumerate(global_accuracy_hist.items(), 1):
        plt.plot(range(1, len(acc) + 1), acc, label=f"Client {idx}")
    
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Global Accuracy", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(VISUAL_DIR, exist_ok=True)
    save_path = os.path.join(VISUAL_DIR, "global_accuracy_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Global accuracy plot saved at: {save_path}")

def visualize_local_accuracy_clients(local_accuracy_hist, VISUAL_DIR):
    """
    Plots local accuracy for each client and saves the figure as a PNG.
    
    Parameters:
        local_accuracy_hist (dict): Dictionary with client IDs as keys and accuracy lists as values.
        VISUAL_DIR: Directory where the PNG file will be saved.
    """
    plt.figure(figsize=(10, 5))
    for idx, (client, acc) in enumerate(local_accuracy_hist.items(), 1):
        plt.plot(range(1, len(acc) + 1), acc, label=f"Client {idx}")
    
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Local Accuracy", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(VISUAL_DIR, exist_ok=True)
    save_path = os.path.join(VISUAL_DIR, "local_accuracy_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Local accuracy plot saved at: {save_path}")