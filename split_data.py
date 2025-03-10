import os
import pandas as pd
import numpy as np
import shutil


names = ["blocker", "ago", "e-"]
shots_list = [10, 20, 40]
split_nums = [0, 1, 2, 3, 4]

# names = ["blocker"]
# shots_list = [10]
# split_nums = [0]

model_path = "var_models"
pkl_path = "var_pkls"

# Data source directory
data_source_dir = "var_data"

# Ensure data source directory exists
if not os.path.exists(data_source_dir):
    print(f"Error: Data source directory {data_source_dir} does not exist. Please run the previous script to generate merged data first.")
    exit(1)

# Check reference folder, get sample counts
reference_folder = "ago_10shots_0"
if os.path.exists(reference_folder):
    # Calculate sample counts
    train_count = len(pd.read_csv(os.path.join(reference_folder, "train.tsv"), sep='\t', header=None))
    valid_count = len(pd.read_csv(os.path.join(reference_folder, "valid.tsv"), sep='\t', header=None))
    test_count = len(pd.read_csv(os.path.join(reference_folder, "test.tsv"), sep='\t', header=None))
    print(f"Sample counts in reference folder: train={train_count}, valid={valid_count}, test={test_count}")
else:
    # Default sample counts
    print(f"Warning: Reference folder {reference_folder} does not exist, using default sample counts")
    train_count = 10  # Default is 10, because it's 10shots
    valid_count = 20  # Observed valid is usually 2x train
    test_count = None  # test will use remaining all data

# Perform data splitting
for name in names:
    # Check if source files exist
    pos_file = os.path.join(data_source_dir, f"{name}.tsv")
    neg_file = os.path.join(data_source_dir, f"{name}_neg.tsv")
    
    if not os.path.exists(pos_file) or not os.path.exists(neg_file):
        print(f"Warning: {pos_file} or {neg_file} does not exist, skipping {name}")
        continue
    
    # Read data
    pos_data = pd.read_csv(pos_file, sep='\t', header=None)
    neg_data = pd.read_csv(neg_file, sep='\t', header=None)
    
    # Ensure positive and negative sample counts are equal
    min_samples = min(len(pos_data), len(neg_data))
    if len(pos_data) != len(neg_data):
        print(f"Warning: {name} positive and negative sample counts are different (pos:{len(pos_data)}, neg:{len(neg_data)}), will truncate to smaller value: {min_samples}")
        pos_data = pos_data.iloc[:min_samples]
        neg_data = neg_data.iloc[:min_samples]
    
    for shots in shots_list:
        # Train set sample counts
        actual_train_count = shots
        
        # Validation set fixed to 20 samples
        actual_valid_count = 20
        
        # Ensure data quantity is enough
        total_required = actual_train_count + actual_valid_count
        if total_required > len(pos_data):
            print(f"Warning: {name} data quantity is not enough, need {total_required} samples, but only {len(pos_data)} samples, skipping {shots}shots")
            continue
        
        for split_num in split_nums:
            # Set random seed
            np.random.seed(split_num)
            
            # Create output directory
            folder_name = f"{name}_{shots}shots_{split_num}"
            output_folder = os.path.join(data_source_dir, folder_name)
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder)
            
            # Shuffle data
            pos_indices = np.random.permutation(len(pos_data))
            neg_indices = np.random.permutation(len(neg_data))
            
            # Split data - take first shots samples for train set
            pos_train = pos_data.iloc[pos_indices[:actual_train_count]]
            neg_train = neg_data.iloc[neg_indices[:actual_train_count]]
            
            # Take next 20 samples for validation set
            pos_valid = pos_data.iloc[pos_indices[actual_train_count:actual_train_count+actual_valid_count]]
            neg_valid = neg_data.iloc[neg_indices[actual_train_count:actual_train_count+actual_valid_count]]
            
            # Test set take remaining all samples
            pos_test = pos_data.iloc[pos_indices[actual_train_count+actual_valid_count:]]
            neg_test = neg_data.iloc[neg_indices[actual_train_count+actual_valid_count:]]
            
            # Save data
            pos_train.to_csv(os.path.join(output_folder, "train.tsv"), sep='\t', index=False, header=False)
            neg_train.to_csv(os.path.join(output_folder, "train_neg.tsv"), sep='\t', index=False, header=False)
            
            pos_valid.to_csv(os.path.join(output_folder, "valid.tsv"), sep='\t', index=False, header=False)
            neg_valid.to_csv(os.path.join(output_folder, "valid_neg.tsv"), sep='\t', index=False, header=False)
            
            pos_test.to_csv(os.path.join(output_folder, "test.tsv"), sep='\t', index=False, header=False)
            neg_test.to_csv(os.path.join(output_folder, "test_neg.tsv"), sep='\t', index=False, header=False)
            
            print(f"Created {folder_name}, train={len(pos_train)}，valid={len(pos_valid)}，test={len(pos_test)}")

            os.makedirs(f"{pkl_path}/{name}_{shots}shots_{split_num}", exist_ok=True)
            os.makedirs(f"{model_path}/{name}_{shots}shots_{split_num}", exist_ok=True)

print("Data splitting completed!") 