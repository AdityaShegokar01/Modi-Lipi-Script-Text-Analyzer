import torch

def safe_torch_load(path, map_location):
    """
    Loads a torch checkpoint using weights_only where supported.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def decode_ctc_output(output, char_map):
    """
    Decodes the raw output from the CTC model into human-readable text.
    Uses a simple best-path decoding.
    """
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
