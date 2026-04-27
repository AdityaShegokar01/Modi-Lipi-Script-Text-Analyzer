import cv2
import numpy as np
import os
from PIL import Image

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")
DEFAULT_BOX_HEIGHT = 10
MIN_LINE_GROUPING_THRESHOLD = 10
LINE_GROUPING_FACTOR = 0.6
_paddleocr_instance = None

def _get_paddleocr():
    global _paddleocr_instance
    if PaddleOCR is None:
        return None
    if _paddleocr_instance is None:
        # Angle classification is disabled because deskewing is handled separately in
        # _deskew_image, which gives more control over rotation thresholds and keeps
        # the detector focused only on line localization.
        _paddleocr_instance = PaddleOCR(use_angle_cls=False, lang=PADDLEOCR_LANG, show_log=False)
    return _paddleocr_instance

def _deskew_image(gray):
    """
    Deskews a grayscale image using the dominant text angle.
    """
    if gray is None or gray.size == 0:
        return gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _extract_line_images_opencv(gray, padding):
    """
    OpenCV-based line extraction from a grayscale image.
    """
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

def _group_boxes_by_line(boxes, image_shape):
    if not boxes:
        return []
    img_h, img_w = image_shape
    box_info = []
    heights = []
    for box in boxes:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x1, x2 = max(0, min(xs)), min(img_w, max(xs))
        y1, y2 = max(0, min(ys)), min(img_h, max(ys))
        h = max(1, y2 - y1)
        heights.append(h)
        box_info.append((x1, y1, x2, y2, (y1 + y2) / 2, h))

    box_info.sort(key=lambda b: b[4])
    median_height = np.median(heights) if heights else DEFAULT_BOX_HEIGHT
    threshold = max(MIN_LINE_GROUPING_THRESHOLD, median_height * LINE_GROUPING_FACTOR)

    line_groups = []
    current_group = [box_info[0]]
    for box in box_info[1:]:
        if abs(box[4] - current_group[-1][4]) <= threshold:
            current_group.append(box)
        else:
            line_groups.append(current_group)
            current_group = [box]
    line_groups.append(current_group)

    line_boxes = []
    for group in line_groups:
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        line_boxes.append((x1, y1, x2, y2))

    line_boxes.sort(key=lambda b: b[1])
    return line_boxes

def _extract_line_images_paddleocr(gray, padding):
    ocr = _get_paddleocr()
    if ocr is None:
        return None

    try:
        # Detection only: no recognition or angle classification for line grouping.
        result = ocr.ocr(gray, det=True, rec=False, cls=False)
    except Exception:
        return None

    if not result or not result[0]:
        return None

    boxes = [entry[0] for entry in result[0]]
    line_boxes = _group_boxes_by_line(boxes, gray.shape)
    if not line_boxes:
        return None

    img_h, img_w = gray.shape
    line_images = []
    for (x1, y1, x2, y2) in line_boxes:
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(img_w, int(x2) + padding)
        y2 = min(img_h, int(y2) + padding)
        crop = gray[y1:y2, x1:x2]
        line_images.append(Image.fromarray(crop))

    return line_images


def extract_line_images(image_input, padding=4, method="auto", deskew=True):
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
        raise TypeError(
            f"Unsupported image type: {type(image_input)}. Expected PIL.Image or numpy.ndarray."
        )

    if deskew:
        gray = _deskew_image(gray)

    line_images = None
    if method in ("auto", "paddleocr"):
        line_images = _extract_line_images_paddleocr(gray, padding)

    if line_images is None:
        line_images = _extract_line_images_opencv(gray, padding)

    return line_images
