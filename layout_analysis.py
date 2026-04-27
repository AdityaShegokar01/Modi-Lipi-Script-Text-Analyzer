"""
layout_analysis.py

A robust OpenCV-based layout analysis module for full-page document images.
Handles denoising, deskewing, adaptive binarization, text line detection,
and line cropping – preparing input for a per-line CRNN OCR model.
"""

import logging
import math

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #

def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (any mode) to an OpenCV BGR uint8 array."""
    rgb = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR uint8 array to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    """Return a single-channel grayscale image from a BGR array."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
#  Preprocessing steps
# --------------------------------------------------------------------------- #

def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means Denoising to reduce paper texture and ink bleed.
    Keeps fine strokes intact while smoothing background noise.
    """
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding produces a clean binary (black-on-white)
    image that handles uneven lighting and shadows across the page.
    Returns a binary image where text pixels are 0 (black) and
    background pixels are 255 (white).
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return binary


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Estimate the dominant text angle using the Hough Line Transform and
    rotate the image so that text lines are horizontal.

    Returns the deskewed grayscale image.
    """
    # Invert so text is white on black for line detection
    inverted = cv2.bitwise_not(gray)

    # Detect edges
    edges = cv2.Canny(inverted, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        logger.debug("Deskew: no lines found; skipping rotation.")
        return gray

    # Collect angles of detected lines (in degrees, relative to horizontal)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Only consider lines that are close to horizontal (±30°)
        if abs(angle) < 30:
            angles.append(angle)

    if not angles:
        logger.debug("Deskew: no near-horizontal lines found; skipping rotation.")
        return gray

    median_angle = float(np.median(angles))
    logger.debug(f"Deskew: rotating by {-median_angle:.2f}°")

    if abs(median_angle) < 0.5:
        return gray  # Negligible skew

    h, w = gray.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(
        gray, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed


# --------------------------------------------------------------------------- #
#  Text-line detection
# --------------------------------------------------------------------------- #

def _detect_line_boxes(binary: np.ndarray, original_h: int, original_w: int):
    """
    Use morphological dilation to merge individual characters into text-line
    blobs, then find their bounding boxes sorted top-to-bottom.

    Parameters
    ----------
    binary : np.ndarray
        Binary image (text=0, background=255) from `binarize()`.
    original_h, original_w : int
        Dimensions of the original image (used to scale the dilation kernel).

    Returns
    -------
    List[Tuple[int, int, int, int]]
        Bounding boxes (x, y, w, h) sorted by their y-coordinate.
    """
    # Invert: text pixels become 255 so morphological ops work naturally
    inverted = cv2.bitwise_not(binary)

    # Horizontal dilation kernel – wide enough to bridge character gaps in a
    # line without merging adjacent lines.  Height of 1 keeps lines separate.
    kernel_w = max(30, original_w // 20)
    kernel_h = max(3, original_h // 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))

    dilated = cv2.dilate(inverted, kernel, iterations=3)

    # Find external contours of the dilated blobs
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    min_area = (original_w * original_h) * 0.0003  # ignore tiny noise blobs
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        # Reject very tall boxes (likely a full-page artefact, not a line)
        if h > original_h * 0.25:
            continue
        boxes.append((x, y, w, h))

    # Sort top-to-bottom (primary), left-to-right (secondary)
    boxes.sort(key=lambda b: (b[1], b[0]))
    logger.debug(f"detect_line_boxes: found {len(boxes)} candidate lines.")
    return boxes


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline: convert → grayscale → denoise → deskew.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The raw uploaded document image (any mode).

    Returns
    -------
    np.ndarray
        A preprocessed single-channel (grayscale) uint8 NumPy array, ready
        for binarization and line detection.
    """
    bgr = _pil_to_bgr(pil_image)
    gray = _to_gray(bgr)
    gray = denoise(gray)
    gray = deskew(gray)
    return gray


def extract_line_crops(pil_image: Image.Image, padding: int = 6):
    """
    Analyse a full-page document image and return a list of cropped line
    images sorted from top to bottom.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The full-page document image (any mode, any size).
    padding : int
        Extra pixels to add around each detected bounding box so that
        ascenders/descenders are not clipped.

    Returns
    -------
    List[PIL.Image.Image]
        Ordered list of PIL grayscale line-crop images.  Returns a list with
        the original image as the sole element if no lines are detected (safe
        fallback for single-line inputs).
    """
    if pil_image is None:
        raise ValueError("pil_image must not be None")

    orig_h, orig_w = pil_image.size[1], pil_image.size[0]

    # Step 1 – Preprocess
    gray = preprocess_image(pil_image)

    # Step 2 – Binarize (on the preprocessed grayscale)
    binary = binarize(gray)

    # Step 3 – Detect line bounding boxes
    boxes = _detect_line_boxes(binary, orig_h, orig_w)

    if not boxes:
        logger.warning(
            "extract_line_crops: no text lines detected; "
            "returning the full image as a single crop."
        )
        # Graceful fallback: treat the entire image as one line
        return [pil_image.convert("L")]

    # Step 4 – Crop with padding, using the preprocessed grayscale image
    crops = []
    for x, y, w, h in boxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(orig_w, x + w + padding)
        y2 = min(orig_h, y + h + padding)

        crop_array = gray[y1:y2, x1:x2]
        if crop_array.size == 0:
            continue

        crop_pil = Image.fromarray(crop_array)
        crops.append(crop_pil)

    if not crops:
        logger.warning(
            "extract_line_crops: all crops were empty; "
            "returning the full image as fallback."
        )
        return [pil_image.convert("L")]

    logger.info(f"extract_line_crops: extracted {len(crops)} line(s).")
    return crops
