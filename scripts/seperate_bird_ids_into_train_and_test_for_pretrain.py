import os
import random
import argparse

def split_bird_ids(input_dir, test_percentage):
    """
    Splits the bird ID directories into training and testing sets based on the specified percentage.

    Args:
        input_dir (str): The directory containing subdirectories of bird IDs.
        test_percentage (float): The percentage of directories to be used for testing.

    Returns:
        tuple: Two lists of directories, one for training and one for testing.
    """
    # Collect all subdirectories (bird IDs) in the input directory
    bird_ids = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # Calculate the number of directories to be used for testing
    total_dirs = len(bird_ids)
    num_test_dirs = int(total_dirs * test_percentage / 100)

    # Shuffle the list of bird IDs
    random.shuffle(bird_ids)

    # Split the shuffled list into test and train sets
    test_dirs = bird_ids[:num_test_dirs]
    train_dirs = bird_ids[num_test_dirs:]

    return train_dirs, test_dirs

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split bird ID directories into training and testing sets.")
    parser.add_argument("input_dir", type=str, help="The directory containing subdirectories of bird IDs.")
    parser.add_argument("test_percentage", type=float, help="The percentage of directories to be used for testing.")
    args = parser.parse_args()

    # Validate the input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist.")
        return

    # Validate the test percentage
    if args.test_percentage < 0 or args.test_percentage > 100:
        print("Error: Test percentage must be a number between 0 and 100.")
        return

    # Split the bird IDs into training and testing sets
    train_dirs, test_dirs = split_bird_ids(args.input_dir, args.test_percentage)

    # Output the lists of directories as space-separated strings
    print(" ".join(train_dirs))
    print(" ".join(test_dirs))

if __name__ == "__main__":
    main()

