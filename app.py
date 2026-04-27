import logging
import os
import codecs
import time
import requests
import json
import torch
import unicodedata
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as T

# Import our custom model and dataset definitions
# These files MUST be in the same folder as app.py
try:
    from model import CRNN
    from dataset import CharacterMap, ResizeAndPad
except ImportError:
    print("="*50)
    print("ERROR: model.py and dataset.py not found.")
    print("Please make sure model.py and dataset.py are in the same folder as app.py")
    print("="*50)
    exit(1)

# Import the layout analysis module for full-page OCR
try:
    from layout_analysis import extract_line_crops
    LAYOUT_ANALYSIS_AVAILABLE = True
except ImportError:
    LAYOUT_ANALYSIS_AVAILABLE = False


# --- Configuration ---
# Model parameters (MUST MATCH train.py)
IMG_HEIGHT = 64
MAX_IMG_WIDTH = 800
INPUT_CHANNELS = 1
RNN_HIDDEN_SIZE = 512

# File paths
MODEL_PATH = os.path.join('models', 'best_model.pth')
CHAR_MAP_PATH = 'char_map.json'
API_KEY_FILE = 'api_key.txt'

# Supported target languages for translation
SUPPORTED_LANGUAGES = [
    "English", "Marathi", "Hindi", "Spanish", "French",
    "German", "Portuguese", "Arabic", "Chinese", "Japanese",
    "Urdu", "Bengali", "Tamil", "Telugu", "Gujarati", "Kannada",
]

# --- App Setup ---
app = Flask(__name__)
CORS(app)  # Allow all origins for simplicity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# --- Global variables for AI models ---
device = None
model = None
char_map = None
gemini_api_key = None
transform = None

def load_ocr_model():
    """
    Loads the trained PyTorch CRNN model and char map into memory.
    """
    global device, model, char_map, transform
    
    try:
        # 1. Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        app.logger.info(f"Loading OCR model on device: {device}")

        # 2. Load CharacterMap
        if not os.path.exists(CHAR_MAP_PATH):
            app.logger.error(f"FATAL: Character map not found at {CHAR_MAP_PATH}")
            return False
            
        with open(CHAR_MAP_PATH, 'r', encoding='utf-8') as f:
            char_map_data = json.load(f)
        
        char_map = CharacterMap()
        char_map.char_to_int = char_map_data['char_to_int']
        char_map.int_to_char = {i: c for c, i in char_map.char_to_int.items()}
        char_map.vocab_size = len(char_map.char_to_int)
        
        vocab_size = char_map.vocab_size
        app.logger.info(f"Character map loaded. Vocab size: {vocab_size}")

        # 3. Initialize Model
        model = CRNN(IMG_HEIGHT, INPUT_CHANNELS, vocab_size, RNN_HIDDEN_SIZE).to(device)
        
        # 4. Load Model Weights
        if not os.path.exists(MODEL_PATH):
            app.logger.error(f"FATAL: Trained model not found at {MODEL_PATH}")
            app.logger.error(f"Please download 'best_model.pth' from Colab and put it in a folder named 'models'")
            return False
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        app.logger.info(f"Trained OCR model loaded from {MODEL_PATH}")

        # 5. Define the transformation pipeline
        transform = T.Compose([
            T.Grayscale(num_output_channels=INPUT_CHANNELS),
            ResizeAndPad(height=IMG_HEIGHT, max_width=MAX_IMG_WIDTH, channels=INPUT_CHANNELS),
            T.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1]
        ])
        
        return True

    except Exception as e:
        app.logger.error(f"An error occurred loading the OCR model: {e}")
        return False

def load_api_key():
    """Loads the Gemini API key from api_key.txt"""
    global gemini_api_key
    try:
        with open(API_KEY_FILE, 'r') as f:
            gemini_api_key = f.read().strip()
        if not gemini_api_key:
            app.logger.warning("api_key.txt is empty.")
            return False
        app.logger.info("Gemini API key loaded successfully.")
        return True
    except FileNotFoundError:
        app.logger.error(f"FATAL: {API_KEY_FILE} not found. Please create it.")
        return False


def _decode_crnn_output(outputs):
    """
    Greedy CTC decode from CRNN output tensor.

    Parameters
    ----------
    outputs : torch.Tensor  shape (seq_len, 1, nclass)

    Returns
    -------
    str  – decoded Devanagari text (NFC-normalised)
    """
    pred_indices = torch.argmax(outputs, dim=2)
    pred_indices = pred_indices.t().cpu().numpy()[0]

    decoded_text = []
    last_char = None
    for idx in pred_indices:
        if idx == 0:          # CTC <BLANK> token
            last_char = None
            continue
        char = char_map.int_to_char.get(idx, '?')
        if char != last_char:
            decoded_text.append(char)
        last_char = char

    return unicodedata.normalize('NFC', "".join(decoded_text))


def predict_ocr(image_file_storage):
    """
    Performs full-page OCR on an uploaded image.

    When the layout_analysis module is available the image is split into
    individual text lines, each line is recognised by the CRNN, and the
    results are joined with newline characters.  If layout analysis is not
    available (import failed) the original single-image path is used as a
    safe fallback.

    Parameters
    ----------
    image_file_storage : werkzeug.datastructures.FileStorage
        The uploaded image file.

    Returns
    -------
    Tuple[str | None, str | None]
        (recognised_text, error_message) – one of the two will be None.
    """
    # --- Open source image ---
    try:
        pil_image = Image.open(image_file_storage)
    except Exception as e:
        app.logger.error(f"Failed to open image: {e}")
        return None, "Invalid image file"

    # --- Decide processing path ---
    if LAYOUT_ANALYSIS_AVAILABLE:
        try:
            line_crops = extract_line_crops(pil_image)
            app.logger.info(f"Layout analysis produced {len(line_crops)} line crop(s).")
        except Exception as e:
            app.logger.warning(
                f"Layout analysis failed ({e}); falling back to single-image mode."
            )
            line_crops = [pil_image.convert("L")]
    else:
        app.logger.warning(
            "layout_analysis module not available; using single-image mode."
        )
        line_crops = [pil_image.convert("L")]

    # --- Run CRNN on every line crop ---
    line_texts = []
    for idx, crop in enumerate(line_crops):
        try:
            image_tensor = transform(crop).to(device)
            image_tensor = image_tensor.unsqueeze(0)   # add batch dim

            with torch.no_grad():
                outputs = model(image_tensor)

            line_text = _decode_crnn_output(outputs)
            if line_text.strip():
                line_texts.append(line_text)
                app.logger.debug(f"  Line {idx + 1}: {line_text!r}")
        except Exception as e:
            app.logger.warning(f"OCR failed on line {idx + 1}: {e}; skipping.")

    if not line_texts:
        app.logger.warning("No text recognised from any line crop.")
        return "", None

    full_text = "\n".join(line_texts)
    app.logger.info(f"OCR complete. {len(line_texts)} line(s) recognised.")
    return full_text, None


# --- Gemini API Helper (with exponential backoff) ---
def fetch_gemini_with_backoff(api_url, payload, retries=5, delay=1):
    headers = {'Content-Type': 'application/json'}
    for i in range(retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=45)
            if response.status_code == 200:
                return response.json()
            else:
                error_text = response.text
                app.logger.error(f"Gemini API returned {response.status_code}. Response: {error_text}")
                if 400 <= response.status_code < 500:
                    return {"error": f"Gemini API client error: {response.status_code}", "details": error_text}
                app.logger.warning(f"Retrying in {delay}s...")
        except requests.exceptions.RequestException as e:
            app.logger.warning(f"Request exception: {e}. Retrying in {delay}s...")
        
        time.sleep(delay)
        delay *= 2
    
    return {"error": "Failed to connect to Gemini API after all retries."}


# --- API Endpoints ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/api/get-text-for-upload', methods=['POST'])
def get_text_for_upload_route():
    """
    Accepts a file upload, runs full-page OCR, and returns the transcribed
    Devanagari text.
    """
    app.logger.info("Received request at /api/get-text-for-upload")
    if 'image_file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files['image_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        png_filename = secure_filename(file.filename)
        app.logger.info(f"Processing uploaded file: {png_filename}")
        
        extracted_text, error_msg = predict_ocr(file)
        
        if error_msg:
            return jsonify({"error": error_msg}), 500
        
        app.logger.info(f"Successfully transcribed text for {png_filename}")
        return jsonify({
            "requested_file": png_filename,
            "content": extracted_text
        }), 200

    return jsonify({"error": "Unknown error processing file upload"}), 500


@app.route('/api/translate', methods=['POST'])
def translate_route():
    """
    Translates the provided Devanagari text (OCR output) into the user's
    chosen target language using the Gemini API.

    Expected JSON body:
    {
        "devanagari_text": "<Devanagari text from OCR>",
        "target_language": "<e.g. English, Marathi, Hindi, Spanish …>"
    }
    """
    app.logger.info("Received request at /api/translate")

    if not gemini_api_key:
        app.logger.error("API Key is missing or was not loaded.")
        return jsonify({"error": "Server is missing Gemini API key"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    devanagari_text = data.get('devanagari_text', '').strip()
    target_language = data.get('target_language', 'English').strip()

    if not devanagari_text:
        return jsonify({"error": "Missing 'devanagari_text'"}), 400

    # Validate / sanitise the target language to avoid prompt injection
    if target_language not in SUPPORTED_LANGUAGES:
        app.logger.warning(
            f"Unsupported target language '{target_language}'; defaulting to English."
        )
        target_language = "English"

    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash-preview-09-2025:generateContent"
        f"?key={gemini_api_key}"
    )

    system_prompt = (
        "You are an expert translator and OCR post-processor specialising in "
        "historical Modi Lipi script and Devanagari text. "
        "You will be given Devanagari text that was produced by an OCR system "
        "and may contain minor spelling errors or misrecognised characters. "
        "Your task is to:\n"
        "1. Contextually correct any obvious OCR spelling mistakes in the "
        "Devanagari text using your knowledge of Marathi and Devanagari "
        "vocabulary.\n"
        f"2. Translate the corrected Devanagari text into {target_language}.\n"
        "3. Return ONLY the final translated text with no extra commentary, "
        "preamble, or explanations."
    )

    user_message = (
        f"Devanagari OCR text to correct and translate into {target_language}:\n\n"
        f"{devanagari_text}"
    )

    payload = {
        "contents": [{"parts": [{"text": user_message}]}],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
    }

    try:
        result = fetch_gemini_with_backoff(api_url, payload)

        if "error" in result:
            app.logger.error(f"Gemini API call failed: {result.get('details', 'Unknown error')}")
            return jsonify({
                "error": result.get("error", "Failed to get response from Gemini"),
                "details": result.get('details'),
            }), 500

        translated_text = result['candidates'][0]['content']['parts'][0]['text']

        if not translated_text:
            return jsonify({"error": "No translation received from model."}), 500

        app.logger.info(f"Successfully translated text to {target_language}.")
        return jsonify({
            "translated_text": translated_text,
            "target_language": target_language,
        }), 200

    except (KeyError, IndexError, TypeError) as e:
        app.logger.error(f"Failed to parse Gemini response: {e}")
        app.logger.error(f"Full Gemini response: {result}")
        return jsonify({"error": "Failed to parse model's answer. See server logs."}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/translate: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


@app.route('/api/ask-gemini', methods=['POST'])
def ask_gemini_route():
    """
    General Q&A endpoint – takes a context and a question and returns
    an answer grounded in that context.
    """
    app.logger.info("Received request at /api/ask-gemini")
    
    if not gemini_api_key:
        app.logger.error("API Key is missing or was not loaded.")
        return jsonify({"error": "Server is missing Gemini API key"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    context = data.get('context')
    question = data.get('question')
    if not context or not question:
        return jsonify({"error": "Missing 'context' or 'question'"}), 400
    
    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash-preview-09-2025:generateContent"
        f"?key={gemini_api_key}"
    )

    system_prompt = (
        "You are a helpful assistant. Answer the user's question based ONLY "
        "on the provided context. If the answer is not found in the context, "
        "state that you cannot find the answer in the provided text. "
        "Do not use any external knowledge."
    )
    user_query = f"Context:\n---\n{context}\n---\n\nQuestion:\n{question}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
    }

    try:
        result = fetch_gemini_with_backoff(api_url, payload)
        
        if "error" in result:
            app.logger.error(f"Gemini API call failed: {result.get('details', 'Unknown error')}")
            return jsonify({"error": result.get("error", "Failed to get response from Gemini"), "details": result.get('details')}), 500

        text = result['candidates'][0]['content']['parts'][0]['text']
        
        if not text:
            return jsonify({"error": "No answer received from model."}), 500
            
        app.logger.info("Successfully got answer from Gemini.")
        return jsonify({"answer": text}), 200

    except (KeyError, IndexError, TypeError) as e:
        app.logger.error(f"Failed to parse Gemini response: {e}")
        app.logger.error(f"Full Gemini response: {result}")
        return jsonify({"error": "Failed to parse model's answer. See server logs."}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred in Gemini route: {str(e)}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


# --- Run Server ---
if __name__ == '__main__':
    print("--- Starting Server ---")
    
    # 1. Load Gemini API Key
    if not load_api_key():
        print("Warning: Could not load Gemini API key. The translation endpoints will fail.")
        
    # 2. Load the custom OCR model
    if not load_ocr_model():
        print("FATAL ERROR: Could not load the OCR model. The server cannot run.")
    else:
        print("--- OCR Model Ready ---")
        if LAYOUT_ANALYSIS_AVAILABLE:
            print("--- Full-Page Layout Analysis: ENABLED ---")
        else:
            print("--- Full-Page Layout Analysis: DISABLED (layout_analysis.py not found) ---")
        print(f"Starting Flask server at http://127.0.0.1:5000")
        print("API is ready to accept file uploads and Gemini requests.")
        # use_reloader=False is important to prevent Flask from loading the model twice
        app.run(debug=True, port=5000, use_reloader=False)


