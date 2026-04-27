import cv2
import numpy as np
from PIL import Image


def extract_line_images(image_input, padding=4):
    """
    Takes a full-page PIL image or numpy array and returns a list of cropped
    line images sorted from top to bottom.

    Steps:
    1. Convert to grayscale numpy array.
    2. Apply Adaptive Binarization to handle uneven lighting/shadows.
    3. Use morphological dilation with a wide horizontal kernel to merge
       individual characters into continuous line blobs.
    4. Find contours to detect bounding boxes for each text line.
    5. Filter out noise and sort the bounding boxes top-to-bottom.
    6. Crop and return each line as a PIL Image.

    Args:
        image_input: A PIL.Image or a numpy ndarray (grayscale or BGR/RGB).
        padding: Extra pixels to add around each detected line crop.

    Returns:
        A list of PIL.Image objects, each containing one line of text,
        sorted from top to bottom. Returns the original image as a single
        element if no lines are detected.
    """
    # --- Step 1: Convert input to a grayscale numpy array ---
    if isinstance(image_input, Image.Image):
        gray = np.array(image_input.convert('L'))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # --- Step 2: Adaptive Binarization ---
    # adaptiveThreshold handles uneven lighting far better than a global threshold.
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # Invert: text = white, bg = black
        blockSize=15,
        C=10,
    )

    # --- Step 3: Morphological dilation with a wide horizontal kernel ---
    # This "smears" nearby characters horizontally so they merge into one blob
    # per text line, while keeping lines vertically separated.
    img_h, img_w = binary.shape
    kernel_w = max(30, img_w // 20)  # 5% of image width, at least 30 px
    kernel_h = 3
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_w, kernel_h)
    )
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # --- Step 4: Find contours ---
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # Fallback: return the original image as a single "line"
        return [Image.fromarray(gray)]

    # --- Step 5: Filter and sort bounding boxes top-to-bottom ---
    bounding_boxes = []
    min_height = max(8, img_h // 80)   # ignore tiny noise blobs
    min_width = max(20, img_w // 40)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_height and w >= min_width:
            bounding_boxes.append((x, y, w, h))

    if not bounding_boxes:
        return [Image.fromarray(gray)]

    # Sort top-to-bottom by the y-coordinate of the bounding box
    bounding_boxes.sort(key=lambda b: b[1])

    # --- Step 6: Crop each line from the grayscale image ---
    line_images = []
    for (x, y, w, h) in bounding_boxes:
        # Add padding, clamped to image boundaries
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        crop = gray[y1:y2, x1:x2]
        line_images.append(Image.fromarray(crop))

    return line_images
