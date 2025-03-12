from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for cross-origin requests

# Create required directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load the model with error handling
def load_ml_model():
    model_path = 'best_model_fold_2.keras'
    try:
        model = load_model(model_path, compile=False)
        logger.info("Model loaded successfully.")
        return model
    except (OSError, IOError) as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        return None

# Initialize model
model = load_ml_model()

# Define labels
labels = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
}

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(380, 380))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict', methods=['POST'])
def predict():
    # Debug request method
    logger.info(f"Received request with method: {request.method}")

    # Ensure POST request
    if request.method != 'POST':
        logger.warning("Invalid method used")
        return jsonify({'error': 'Method Not Allowed. Use POST.'}), 405

    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded. Please check if the model file exists.'})

    # Check for file
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file part in the request.'})

    file = request.files['file']
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No selected file.'})

    if file and allowed_file(file.filename):
        try:
            # Create unique filename
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join('uploads', unique_filename)
            
            # Save file
            file.save(file_path)
            logger.info(f"File saved: {file_path}")
            
            # Preprocess
            img = preprocess_image(file_path)
            
            # Predict
            logger.info("Running prediction")
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = labels[predicted_class]
            confidence = float(prediction[0][predicted_class]) * 100
            
            logger.info(f"Prediction result: {predicted_label}, Confidence: {confidence:.2f}%")
            
            # Return result
            result = {
                'prediction': predicted_label,
                'confidence': f'{confidence:.2f}%',
                'class_id': int(predicted_class),
                'image_path': f'/uploads/{unique_filename}'
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.exception(f"Error in prediction process: {str(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'})
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'})

if __name__ == '__main__':
    # Check model status
    if model is None:
        logger.warning("⚠️ WARNING: Model could not be loaded!")
        logger.warning("The application will start but predictions will fail.")
    
    # Start server
    host = '0.0.0.0'
    port = 8080
    
    logger.info(f"Starting server on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
