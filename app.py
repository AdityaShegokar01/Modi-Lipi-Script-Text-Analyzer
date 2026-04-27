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
    from preprocessing import extract_line_images
except ImportError:
    print("="*50)
    print("ERROR: model.py, dataset.py, or preprocessing.py not found.")
    print("Please make sure these files are in the same folder as app.py")
    print("="*50)
    exit(1)


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

# --- App Setup ---
app = Flask(__name__)
CORS(app)  # Allow all origins for simplicity

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

def _decode_single_line(image_pil):
    """
    Runs the CRNN model on a single PIL line image and returns the decoded text.
    """
    image_tensor = transform(image_pil).to(device)
    image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)

    # outputs shape: (seq_len, batch, nclass)
    pred_indices = torch.argmax(outputs, dim=2)
    pred_indices = pred_indices.t().cpu().numpy()[0]  # first item in batch

    decoded_text = []
    last_char = None
    for idx in pred_indices:
        if idx == 0:  # 0 is the CTC <BLANK> token
            last_char = None
            continue
        char = char_map.int_to_char.get(idx, '?')
        if char != last_char:
            decoded_text.append(char)
        last_char = char

    return "".join(decoded_text)


def predict_ocr(image_file_storage):
    """
    Performs full-page OCR on an uploaded image.

    The image is preprocessed with ``preprocessing.extract_line_images`` to
    detect individual text lines. Each line is passed through the CRNN model
    sequentially and the results are joined with newline characters to produce
    the complete transcription.
    """
    try:
        full_page_image = Image.open(image_file_storage)
    except Exception as e:
        app.logger.error(f"Failed to open image: {e}")
        return None, "Invalid image file"

    try:
        # Extract individual line crops from the full page
        line_images = extract_line_images(full_page_image)
        app.logger.info(f"Detected {len(line_images)} text line(s) in the uploaded image.")

        line_texts = []
        for i, line_img in enumerate(line_images):
            # Ensure grayscale for the model transform
            line_img_gray = line_img.convert('L')
            line_text = _decode_single_line(line_img_gray)
            line_text = unicodedata.normalize('NFC', line_text)
            app.logger.debug(f"Line {i+1}: {line_text!r}")
            line_texts.append(line_text)

        final_text = "\n".join(line_texts)
        return final_text, None

    except Exception as e:
        app.logger.error(f"Error during OCR prediction: {e}")
        return None, "Error during model prediction."

# --- Gemini API Helper (with backoff) ---
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
    # Helper to serve the HTML file
    return send_from_directory('.', 'index.html')

@app.route('/api/get-text-for-upload', methods=['POST'])
def get_text_for_upload_route():
    """
    API endpoint that accepts a file upload, performs OCR, and returns text.
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
        
        # Perform OCR
        extracted_text, error_msg = predict_ocr(file)
        
        if error_msg:
            return jsonify({"error": error_msg}), 500
        
        app.logger.info(f"Successfully transcribed text for {png_filename}")
        return jsonify({
            "requested_file": png_filename,
            "content": extracted_text
        }), 200

    return jsonify({"error": "Unknown error processing file upload"}), 500

@app.route('/api/ask-gemini', methods=['POST'])
def ask_gemini_route():
    """
    API endpoint that takes a context and a question for the Gemini API.
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
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={gemini_api_key}"

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
        print("Warning: Could not load Gemini API key. The /api/ask-gemini endpoint will fail.")
        
    # 2. Load the custom OCR model
    if not load_ocr_model():
        print("FATAL ERROR: Could not load the OCR model. The server cannot run.")
    else:
        print("--- OCR Model Ready ---")
        print(f"Starting Flask server at http://127.0.0.1:5000")
        print("API is ready to accept file uploads and Gemini requests.")
        # use_reloader=False is important to prevent Flask from loading the model twice
        app.run(debug=True, port=5000, use_reloader=False)

