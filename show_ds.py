import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

def compute_answer_lengths(dataset):
    """
    Extracts the length (in words) of the answer from each training example.
    Assumes that the answer is the message with role 'assistant'.
    """
    answer_lengths = []
    # Iterate using integer indexing so each entry is a dict.
    for i in range(len(dataset)):
        entry = dataset[i]
        # Look for the assistant message
        if "messages" in entry:
            for message in entry["messages"]:
                if message["role"] == "assistant":
                    answer = message["content"]
                    # Calculate number of words
                    length = len(answer.split())
                    answer_lengths.append(length)
                    break  # Only one assistant message per entry is expected
        else:
            print(f"Warning: entry {i} does not contain 'messages'")
    return np.array(answer_lengths)

def print_statistics(answer_lengths):
    """
    Computes and prints basic statistics for answer lengths.
    """
    mean_length = np.mean(answer_lengths)
    median_length = np.median(answer_lengths)
    std_length = np.std(answer_lengths)
    min_length = np.min(answer_lengths)
    max_length = np.max(answer_lengths)
    
    print("Answer Length Statistics:")
    print(f"Mean: {mean_length:.2f} words")
    print(f"Median: {median_length} words")
    print(f"Standard Deviation: {std_length:.2f} words")
    print(f"Minimum Length: {min_length} words")
    print(f"Maximum Length: {max_length} words")

def plot_histogram(answer_lengths, bins=None):
    """
    Plots a histogram of the answer lengths.
    """
    if bins is None:
        # Create bins with a width of 5 words
        bins = range(0, max(answer_lengths) + 5, 5)
    
    plt.figure()
    plt.hist(answer_lengths, bins=bins, edgecolor="black")
    plt.xlabel("Answer Length (words)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Answer Lengths")
    plt.show()

def display_sample_entries(dataset, n_samples=1):
    """
    Displays a few sample entries from the training dataset randomly.
    """
    total_entries = len(dataset)
    sample_indices = random.sample(range(total_entries), n_samples)
    print("\nRandom Sample Entries from Training Dataset:")
    for i, idx in enumerate(sample_indices):
        entry = dataset[idx]  # Use integer indexing to get a row (dict)
        print(f"\nEntry {i + 1} (index {idx}):")
        if isinstance(entry, dict) and "messages" in entry:
            for message in entry["messages"]:
                role = message["role"]
                content = message["content"]
                print(f"{role.capitalize()}: {content}")
        else:
            print("Entry format unexpected:", entry)

def main():
    # Path where the training dataset is saved.
    dataset_path = os.path.join("data", "datasets", "sd_dataset")
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Compute answer lengths and display statistics.
    #answer_lengths = compute_answer_lengths(dataset)
    #print_statistics(answer_lengths)
    #plot_histogram(answer_lengths)
    
    # Display sample entries.
    display_sample_entries(dataset, n_samples=3)

if __name__ == "__main__":
    main()
