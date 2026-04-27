import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

# Import our custom modules
from dataset import OCRDataset, CharacterMap, collate_fn
from model import CRNN

# --- Configuration ---
# Data paths
TRAIN_GT_FILE = 'train.txt'
VALIDATION_GT_FILE = 'validation.txt'
CHAR_MAP_FILE = 'char_map.json'
MODEL_SAVE_DIR = 'models'
BEST_MODEL_NAME = 'best_model.pth'

# Model parameters (MUST MATCH model.py and dataset.py)
IMG_HEIGHT = 64
MAX_IMG_WIDTH = 800 # Set a fixed max width for padding
INPUT_CHANNELS = 1 # Grayscale
RNN_HIDDEN_SIZE = 512

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
# --- MODIFIED: Lowered LR for stability ---
LEARNING_RATE = 0.0001 # 1e-4
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune the CRNN OCR model.")
    parser.add_argument("--train-file", default=TRAIN_GT_FILE, help="Path to training ground-truth file.")
    parser.add_argument("--val-file", default=VALIDATION_GT_FILE, help="Path to validation ground-truth file.")
    parser.add_argument("--char-map", default=CHAR_MAP_FILE, help="Path to the character map JSON.")
    parser.add_argument("--model-dir", default=MODEL_SAVE_DIR, help="Directory to save model checkpoints.")
    parser.add_argument("--pretrained", default=None, help="Optional path to a pretrained model for fine-tuning.")
    parser.add_argument("--rebuild-char-map", action="store_true", help="Rebuild char_map.json from training data.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentations.")
    return parser.parse_args()

def load_pretrained_weights(model, model_path):
    if not model_path:
        return
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(model, nn.DataParallel):
        model_to_load = model.module
    else:
        model_to_load = model

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_to_load.load_state_dict(state_dict, strict=True)
    print(f"Loaded pretrained weights from {model_path}")

def decode_ctc_output(output, char_map):
    """
    Decodes the raw output from the CTC model into human-readable text.
    Uses a simple best-path decoding.
    """
    # output shape: (seq_len, batch_size, num_classes)
    pred_indices = torch.argmax(output, dim=2)
    # pred_indices shape: (seq_len, batch_size)
    
    # Transpose to (batch_size, seq_len)
    pred_indices = pred_indices.t().cpu().numpy()
    
    decoded_texts = []
    
    for indices in pred_indices:
        decoded_text = []
        last_char = None
        for idx in indices:
            if idx == 0: # 0 is the CTC <BLANK> token
                last_char = None
                continue
            
            char = char_map.int_to_char.get(idx, '?')
            
            if char != last_char:
                decoded_text.append(char)
            last_char = char
            
        decoded_texts.append("".join(decoded_text))
    return decoded_texts

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Prepare Data ---
    print("Building character map...")
    if args.rebuild_char_map or not os.path.exists(args.char_map):
        char_map_builder = CharacterMap(args.train_file, args.char_map)
    else:
        char_map_builder = CharacterMap() # Create empty map
        char_map_builder.load_map(args.char_map) # Load from file
        print(f"Character map loaded from {args.char_map} (vocab size: {char_map_builder.vocab_size})")
        
    vocab_size = char_map_builder.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = OCRDataset(
        args.train_file,
        char_map_builder,
        IMG_HEIGHT,
        MAX_IMG_WIDTH,
        augment=not args.no_augment
    )
    val_dataset = OCRDataset(args.val_file, char_map_builder, IMG_HEIGHT, MAX_IMG_WIDTH)

    # Use num_workers=2 for Colab
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print("Data loading complete.")

    # --- 2. Initialize Model, Loss, Optimizer ---
    print("Initializing model...")
    model = CRNN(IMG_HEIGHT, INPUT_CHANNELS, vocab_size, RNN_HIDDEN_SIZE).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    load_pretrained_weights(model, args.pretrained)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Scheduler to lower LR if training gets stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=5, 
        factor=0.1
    )

    os.makedirs(args.model_dir, exist_ok=True)
    best_model_path = os.path.join(args.model_dir, BEST_MODEL_NAME)
    
    best_val_loss = float('inf')

    # --- 3. Training Loop ---
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for i, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            input_lengths = torch.full(
                size=(outputs.size(1),), 
                fill_value=outputs.size(0), 
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            
            # Add Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                outputs = model(images)
                input_lengths = torch.full(
                    size=(outputs.size(1),), 
                    fill_value=outputs.size(0), 
                    dtype=torch.long
                ).to(device)
                
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Average Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Show Example Decoding
        try:
            images_ex, targets_ex, target_lengths_ex = next(iter(val_loader))
            images_ex = images_ex.to(device)
            
            outputs_ex = model(images_ex)
            decoded = decode_ctc_output(outputs_ex, char_map_builder)
            
            print("\n--- Validation Example ---")
            print(f"  Prediction: {decoded[0]}")
            
            # --- FIX for "cannot be converted to Scalar" error ---
            # Get the length of the first target as a Python integer
            first_len = target_lengths_ex[0].item()
            
            # Get the first target's indices
            target_indices = targets_ex[0][:first_len]
            
            # Loop through the 0D tensors and convert to text
            true_text = "".join([char_map_builder.int_to_char.get(t.item(), '?') for t in target_indices])
            # --- END FIX ---

            print(f"  Ground Truth: {true_text}")
            print("--------------------------\n")
        except Exception as e:
            print(f"Error in decoding example: {e}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"*** New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f}) ***\n")

    print("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    train(args)

