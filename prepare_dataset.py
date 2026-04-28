import os
import random
import json
import unicodedata

# --- Configuration ---
# Set the path to your single dataset folder
DATASET_DIR = 'Dataset' 
IMAGE_EXTENSION = '.png'
# Set your specific text file extension
TEXT_EXTENSION = '.gt.txt' 

# Define the split ratios
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
# TEST_RATIO will be what's left (approx 0.1)

# Output filenames
TRAIN_FILE = 'train.txt'
VALIDATION_FILE = 'validation.txt'
TEST_FILE = 'test.txt'
CHAR_MAP_FILE = 'char_map.json'
# ---------------------

def normalize_text(text):
    """
    Normalizes Unicode text to a consistent representation (NFC).
    This prevents issues where the same character has multiple byte representations.
    """
    return unicodedata.normalize('NFC', text)

def build_dataset_files():
    print(f"Scanning directory: {DATASET_DIR}...")
    all_file_pairs = []
    charset = set()
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(DATASET_DIR):
        for file_name in files:
            if file_name.endswith(IMAGE_EXTENSION):
                img_path = os.path.join(root, file_name)
                
                # Get the base name (e.g., 'image_001')
                base_name = os.path.splitext(file_name)[0]
                
                # Find the corresponding text file
                text_file_name = base_name + TEXT_EXTENSION
                text_file_path = os.path.join(root, text_file_name)
                
                if os.path.exists(text_file_path):
                    try:
                        with open(text_file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if text:
                            # Normalize text and add to our lists
                            normalized_text = normalize_text(text)
                            all_file_pairs.append(f"{img_path}|{normalized_text}")
                            charset.update(normalized_text)
                        else:
                            print(f"Warning: Empty text file: {text_file_path}")
                            
                    except Exception as e:
                        print(f"Error reading {text_file_path}: {e}")
                else:
                    print(f"Warning: No matching text file found for {img_path}")

    if not all_file_pairs:
        print("Error: No valid image/text pairs were found. Please check your folder and extensions.")
        return

    print(f"Found {len(all_file_pairs)} valid image/text pairs.")

    # Shuffle the dataset
    random.shuffle(all_file_pairs)

    # Split the dataset
    total_count = len(all_file_pairs)
    train_count = int(total_count * TRAIN_RATIO)
    validation_count = int(total_count * VALIDATION_RATIO)
    
    train_pairs = all_file_pairs[:train_count]
    validation_pairs = all_file_pairs[train_count : train_count + validation_count]
    test_pairs = all_file_pairs[train_count + validation_count :]

    # Write the files
    try:
        with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(train_pairs))
        print(f"Created {TRAIN_FILE} with {len(train_pairs)} entries.")

        with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(validation_pairs))
        print(f"Created {VALIDATION_FILE} with {len(validation_pairs)} entries.")

        with open(TEST_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(test_pairs))
        print(f"Created {TEST_FILE} with {len(test_pairs)} entries.")

    except IOError as e:
        print(f"Error writing split files: {e}")
        return

    # --- Build and Save Character Map ---
    print("\nBuilding character map...")
    # CTC Blank token is 0
    char_to_int = {"<BLANK>": 0}
    int_to_char = {0: "<BLANK>"}
    
    # Start other characters from 1
    for i, char in enumerate(sorted(list(charset)), 1):
        char_to_int[char] = i
        int_to_char[i] = char
        
    vocab_size = len(char_to_int)
    print(f"Built character map with {vocab_size} characters (including CTC blank).")

    try:
        with open(CHAR_MAP_FILE, 'w', encoding='utf-8') as f:
            json.dump({'char_to_int': char_to_int}, f, ensure_ascii=False, indent=4)
        print(f"Character map saved to {CHAR_MAP_FILE}")
    except IOError as e:
        print(f"Error writing character map: {e}")

if __name__ == "__main__":
    build_dataset_files()

