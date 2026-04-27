import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from PIL import Image
import os
import json
import unicodedata

GAUSSIAN_NOISE_MEAN = 0.0
GAUSSIAN_NOISE_STD = 0.05
GAUSSIAN_NOISE_PROB = 0.3
AUGMENT_BLUR_PROB = 0.3
AUGMENT_BLUR_SIGMA = (0.1, 2.0)
AUGMENT_COLORJITTER_PROB = 0.3
AUGMENT_COLORJITTER_BRIGHTNESS = 0.3
AUGMENT_COLORJITTER_CONTRAST = 0.3
AUGMENT_AFFINE_PROB = 0.3
AUGMENT_AFFINE_DEGREES = 2
AUGMENT_AFFINE_TRANSLATE = (0.02, 0.02)
AUGMENT_AFFINE_SHEAR = 2
AUGMENT_PERSPECTIVE_PROB = 0.3
AUGMENT_PERSPECTIVE_DISTORTION = 0.2

class CharacterMap:
    """
    Manages the mapping between characters and integers.
    """
    def __init__(self, gt_file=None, save_path=None):
        self.char_to_int = {"<BLANK>": 0}
        self.int_to_char = {0: "<BLANK>"}
        self.vocab_size = 1
        if gt_file and save_path:
            self.build_map(gt_file, save_path)

    def build_map(self, gt_file, save_path):
        charset = set()
        print(f"Building character map from: {gt_file}")
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        _, text = line.strip().split('|', 1)
                        charset.update(unicodedata.normalize('NFC', text))
                    except ValueError:
                        print(f"Warning: Skipping malformed line: {line.strip()}")
            
            for i, char in enumerate(sorted(list(charset)), 1):
                self.char_to_int[char] = i
                self.int_to_char[i] = i
            
            self.vocab_size = len(self.char_to_int)
            self.save_map(save_path)
            
        except FileNotFoundError:
            print(f"Error: Ground truth file not found at {gt_file}")
            raise
        except Exception as e:
            print(f"Error building character map: {e}")
            raise

    def save_map(self, save_path):
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({'char_to_int': self.char_to_int}, f, ensure_ascii=False, indent=4)
            print(f"Character map saved to {save_path} (vocab size: {self.vocab_size})")
        except IOError as e:
            print(f"Error writing character map: {e}")

    def load_map(self, map_path):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.char_to_int = data['char_to_int']
            self.int_to_char = {i: c for c, i in self.char_to_int.items()}
            self.vocab_size = len(self.char_to_int)
        except FileNotFoundError:
            print(f"Error: Character map file not found at {map_path}")
            raise
        except Exception as e:
            print(f"Error loading character map: {e}")
            raise

    def text_to_indices(self, text):
        return [self.char_to_int.get(char, 0) for char in unicodedata.normalize('NFC', text)]

    def indices_to_text(self, indices):
        return "".join([self.int_to_char.get(idx, '?') for idx in indices])


class ResizeAndPad:
    """
    Custom transform to resize image to fixed height and pad to max width.
    """
    def __init__(self, height, max_width, channels=1):
        self.height = height
        self.max_width = max_width
        self.channels = channels
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        # 1. Resize to fixed height, maintain aspect ratio
        w, h = img.size
        aspect_ratio = w / h
        new_w = int(self.height * aspect_ratio)
        
        # Use ANTIALIAS for better quality resizing
        img = img.resize((new_w, self.height), Image.Resampling.LANCZOS)
        
        # 2. Convert to tensor
        img_tensor = self.to_tensor(img)
        
        # 3. Pad to max_width
        _, h_tensor, w_tensor = img_tensor.size()
        
        if w_tensor > self.max_width:
            # If wider, crop from the right
            img_tensor = img_tensor[:, :, :self.max_width]
            pad_width = 0
        else:
            # If narrower, pad on the right
            pad_width = self.max_width - w_tensor
        
        # Pad (left, right, top, bottom)
        padding = (0, pad_width, 0, 0)
        padded_tensor = torch.nn.functional.pad(img_tensor, padding, "constant", 0) # Pad with 0
        
        return padded_tensor

class AddGaussianNoise:
    """
    Adds gaussian noise to a tensor image.

    Args:
        mean: Mean of the gaussian noise (default 0.0).
        std: Standard deviation of the gaussian noise (default 0.05).
        p: Probability of applying the noise (default 0.3).
    """
    def __init__(self, mean=0.0, std=0.05, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() >= self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0.0, 1.0)

class OCRDataset(Dataset):
    """
    PyTorch Dataset for loading OCR data.
    """
    def __init__(self, gt_file, char_map, img_height, max_img_width, channels=1, augment=False):
        self.gt_file = gt_file
        self.char_map = char_map
        self.lines = []
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                self.lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Ground truth file not found at {gt_file}")
            raise

        augment_transforms = []
        tensor_augment_transforms = []
        if augment:
            augment_transforms = [
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=AUGMENT_BLUR_SIGMA)], p=AUGMENT_BLUR_PROB),
                T.RandomApply(
                    [T.ColorJitter(brightness=AUGMENT_COLORJITTER_BRIGHTNESS, contrast=AUGMENT_COLORJITTER_CONTRAST)],
                    p=AUGMENT_COLORJITTER_PROB
                ),
                T.RandomApply(
                    [T.RandomAffine(
                        degrees=AUGMENT_AFFINE_DEGREES,
                        translate=AUGMENT_AFFINE_TRANSLATE,
                        shear=AUGMENT_AFFINE_SHEAR,
                        fill=0
                    )],
                    p=AUGMENT_AFFINE_PROB
                ),
                T.RandomPerspective(distortion_scale=AUGMENT_PERSPECTIVE_DISTORTION, p=AUGMENT_PERSPECTIVE_PROB),
            ]
            tensor_augment_transforms = [
                AddGaussianNoise(mean=GAUSSIAN_NOISE_MEAN, std=GAUSSIAN_NOISE_STD, p=GAUSSIAN_NOISE_PROB)
            ]

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=channels),
            *augment_transforms,
            ResizeAndPad(height=img_height, max_width=max_img_width, channels=channels),
            *tensor_augment_transforms,
            T.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        try:
            line = self.lines[idx]
            img_path, text = line.split('|', 1)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found {img_path}")
                # Return a dummy tensor if file is missing
                return self.__getitem__((idx + 1) % len(self)) 
            
            image = Image.open(img_path).convert('L') # Convert to 'L' (grayscale)
            
            if self.transform:
                image = self.transform(image)
                
            target = torch.tensor(self.char_map.text_to_indices(text), dtype=torch.long)
            
            return image, target
        
        except Exception as e:
            print(f"Error loading item at index {idx} ({self.lines[idx]}): {e}")
            # Try to load next item to avoid crash
            return self.__getitem__((idx + 1) % len(self))


def collate_fn(batch):
    """
    Custom collate function to pad targets in a batch.
    """
    images, targets = zip(*batch)
    
    # Stack images (they are already padded to same size)
    images = torch.stack(images, 0)
    
    # Get lengths of targets
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    
    # Pad targets
    targets = pad_sequence(targets, batch_first=True, padding_value=0) # Use 0 for <BLANK>
    
    return images, targets, target_lengths

