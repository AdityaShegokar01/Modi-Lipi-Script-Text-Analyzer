import argparse
import os
import unicodedata
import torch
from torch.utils.data import DataLoader

from dataset import OCRDataset, CharacterMap, collate_fn
from model import CRNN

# Model parameters (MUST MATCH model.py and dataset.py)
IMG_HEIGHT = 64
MAX_IMG_WIDTH = 800
INPUT_CHANNELS = 1
RNN_HIDDEN_SIZE = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the CRNN OCR model with CER/WER.")
    parser.add_argument("--test-file", default="test.txt", help="Path to test ground-truth file.")
    parser.add_argument("--char-map", default="char_map.json", help="Path to the character map JSON.")
    parser.add_argument("--model-path", default=os.path.join("models", "best_model.pth"), help="Path to model weights.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    return parser.parse_args()

def decode_ctc_output(output, char_map):
    pred_indices = torch.argmax(output, dim=2)
    pred_indices = pred_indices.t().cpu().numpy()

    decoded_texts = []
    for indices in pred_indices:
        decoded_text = []
        last_char = None
        for idx in indices:
            if idx == 0:
                last_char = None
                continue
            char = char_map.int_to_char.get(idx, '?')
            if char != last_char:
                decoded_text.append(char)
            last_char = char
        decoded_texts.append("".join(decoded_text))
    return decoded_texts

def levenshtein_distance(seq_a, seq_b):
    if seq_a == seq_b:
        return 0
    if len(seq_a) == 0:
        return len(seq_b)
    if len(seq_b) == 0:
        return len(seq_a)

    previous = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        current = [i]
        for j, item_b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if item_a == item_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

def normalize_text(text):
    return unicodedata.normalize('NFC', text).strip()

def evaluate():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.char_map):
        raise FileNotFoundError(f"Character map not found: {args.char_map}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")

    char_map = CharacterMap()
    char_map.load_map(args.char_map)

    dataset = OCRDataset(
        args.test_file,
        char_map,
        IMG_HEIGHT,
        MAX_IMG_WIDTH,
        augment=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = CRNN(IMG_HEIGHT, INPUT_CHANNELS, char_map.vocab_size, RNN_HIDDEN_SIZE).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    total_char_edits = 0
    total_char_len = 0
    total_word_edits = 0
    total_word_len = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = decode_ctc_output(outputs, char_map)

            for i, pred in enumerate(preds):
                target_len = target_lengths[i].item()
                target_indices = targets[i][:target_len].tolist()
                truth = char_map.indices_to_text(target_indices)

                pred = normalize_text(pred)
                truth = normalize_text(truth)

                total_char_edits += levenshtein_distance(list(pred), list(truth))
                total_char_len += len(truth)

                pred_words = pred.split()
                truth_words = truth.split()
                total_word_edits += levenshtein_distance(pred_words, truth_words)
                total_word_len += len(truth_words)

    cer = total_char_edits / max(total_char_len, 1)
    wer = total_word_edits / max(total_word_len, 1)

    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")

if __name__ == "__main__":
    evaluate()
