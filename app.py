from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np
import mimetypes
from collections import Counter
import math
import urllib.request  # Added for downloading model

app = Flask(__name__)
CORS(app)

# ---------------------------
# 🔧 Configuration
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, '../temp')
WEB_DIR = os.path.join(BASE_DIR, '../web')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# 🌐 Download YOLO model from Azure Blob Storage if not exists
# ---------------------------
MODEL_URL = "https://ecoscanyolo.blob.core.windows.net/models/last1.pt"
MODEL_PATH = os.path.join(MODEL_DIR, "last1.pt")

if not os.path.exists(MODEL_PATH):
    print(f"📥 Downloading YOLO model from Blob Storage...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"✅ Model downloaded to: {MODEL_PATH}")
else:
    print(f"✅ Model already exists: {MODEL_PATH}")

# Load trained YOLOv8 model
model = YOLO(MODEL_PATH)

# Define biodegradable and non-biodegradable categories
BIODEGRADABLE_LABELS = ['BIODEGRADABLE', 'PAPER', 'CARDBOARD']
NON_BIODEGRADABLE_LABELS = ['GLASS', 'METAL', 'PLASTIC']

# Energy generation potential (kWh per kg) for different materials
ENERGY_POTENTIAL = {
    'BIODEGRADABLE': 0.8,    # Anaerobic digestion
    'PAPER': 0.6,            # Incineration
    'CARDBOARD': 0.5,        # Incineration
    'GLASS': 0.0,            # No energy generation
    'METAL': 0.0,            # No energy generation (but recyclable)
    'PLASTIC': 2.5           # Incineration (high energy)
}

# Material density (kg/m³) for volume to mass conversion
MATERIAL_DENSITY = {
    'BIODEGRADABLE': 400,    # Organic waste
    'PAPER': 800,            # Paper
    'CARDBOARD': 600,        # Cardboard
    'GLASS': 2500,           # Glass
    'METAL': 7800,           # Steel
    'PLASTIC': 950           # Plastic
}

# Real-world object size references (cm) for different materials
REAL_WORLD_SIZES = {
    'BIODEGRADABLE': {'min': 2, 'max': 15, 'typical': 8},      # Organic waste pieces
    'PAPER': {'min': 5, 'max': 30, 'typical': 15},             # Paper sheets
    'CARDBOARD': {'min': 10, 'max': 50, 'typical': 25},        # Cardboard boxes
    'GLASS': {'min': 3, 'max': 20, 'typical': 10},             # Glass bottles/containers
    'METAL': {'min': 5, 'max': 25, 'typical': 12},             # Metal cans/objects
    'PLASTIC': {'min': 3, 'max': 25, 'typical': 12}            # Plastic bottles/containers
}

# Confidence threshold for detections (adjust this value)
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for your current model performance

# Default real-world area assumption used when no calibration is provided (cm²)
DEFAULT_SCENE_AREA_CM2 = 600.0

# Preprocessing configuration
ENABLE_AUTOMATIC_SCALE_CORRECTION = True
ENABLE_SIZE_VALIDATION = True
ENABLE_PERSPECTIVE_CORRECTION = True
MAX_SCALE_FACTOR = 10.0
MIN_SCALE_FACTOR = 0.1

# ---------------------------
# 🖼️ Image Preprocessing Functions
# ---------------------------
def estimate_object_scale_factor(image, detections, label_counts):
    if not detections or not label_counts:
        return 1.0
    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
    typical_size_cm = REAL_WORLD_SIZES.get(most_common_label, {}).get('typical', 10)
    largest_detection = None
    max_area = 0
    for detection in detections:
        if detection['label'] == most_common_label:
            if detection['area_pixels'] > max_area:
                max_area = detection['area_pixels']
                largest_detection = detection
    if largest_detection:
        detected_size_cm = math.sqrt(largest_detection['area_cm2'])
        if detected_size_cm > 0:
            scale_factor = typical_size_cm / detected_size_cm
            scale_factor = max(0.1, min(10.0, scale_factor))
            return scale_factor
    return 1.0

def apply_perspective_correction(image):
    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    margin = min(width, height) * 0.1
    dst_points = np.float32([[margin, margin], [width - margin, margin], [width - margin, height - margin], [margin, height - margin]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_image = cv2.warpPerspective(image, matrix, (width, height))
    return corrected_image

def normalize_object_sizes(detections, scale_factor):
    normalized_detections = []
    for detection in detections:
        corrected_area_cm2 = detection['area_cm2'] * (scale_factor ** 2)
        corrected_mass_kg = detection['mass_kg'] * (scale_factor ** 2)
        corrected_energy_kwh = detection['energy_potential_kwh'] * (scale_factor ** 2)
        normalized_detection = detection.copy()
        normalized_detection['area_cm2'] = round(corrected_area_cm2, 2)
        normalized_detection['mass_kg'] = round(corrected_mass_kg, 4)
        normalized_detection['energy_potential_kwh'] = round(corrected_energy_kwh, 4)
        normalized_detection['scale_correction_applied'] = True
        normalized_detection['scale_factor'] = round(scale_factor, 3)
        normalized_detections.append(normalized_detection)
    return normalized_detections

def validate_object_sizes(detections):
    validated_detections = []
    for detection in detections:
        label = detection['label']
        area_cm2 = detection['area_cm2']
        size_constraints = REAL_WORLD_SIZES.get(label, {'min': 1, 'max': 100, 'typical': 20})
        min_size_cm = size_constraints['min']
        max_size_cm = size_constraints['max']
        current_size_cm = math.sqrt(area_cm2)
        if current_size_cm < min_size_cm or current_size_cm > max_size_cm:
            typical_size_cm = size_constraints['typical']
            typical_area_cm2 = typical_size_cm ** 2
            material_density = MATERIAL_DENSITY.get(label, 1000)
            volume_m3 = (typical_area_cm2 / 10000.0) * 0.01
            corrected_mass_kg = volume_m3 * material_density
            corrected_energy_kwh = corrected_mass_kg * ENERGY_POTENTIAL.get(label, 0)
            detection['area_cm2'] = round(typical_area_cm2, 2)
            detection['mass_kg'] = round(corrected_mass_kg, 4)
            detection['energy_potential_kwh'] = round(corrected_energy_kwh, 4)
            detection['size_corrected'] = True
            detection['original_size_cm'] = round(current_size_cm, 2)
            detection['corrected_size_cm'] = typical_size_cm
        validated_detections.append(detection)
    return validated_detections

# ---------------------------
# 🔍 Predict endpoint
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # ... keep all your original predict code unchanged ...
    return jsonify({'message': 'Predict endpoint unchanged'})  # Placeholder

# ---------------------------
# ⚡ Energy Efficiency Calculation
# ---------------------------
def calculate_energy_efficiency(total_energy_kwh, total_mass_kg):
    total_energy_kwh = float(total_energy_kwh)
    total_mass_kg = float(total_mass_kg)
    if total_mass_kg <= 0:
        return 0.0
    max_theoretical_energy = float(total_mass_kg) * float(max(ENERGY_POTENTIAL.values()))
    if max_theoretical_energy <= 0:
        return 0.0
    efficiency = (float(total_energy_kwh) / float(max_theoretical_energy)) * 100.0
    return float(round(efficiency, 2))

# ---------------------------
# 🌱 Biodegradability Analysis
# ---------------------------
def analyze_biodegradability(label_counts):
    # ... keep original code unchanged ...
    return {}

# ---------------------------
# 🖼️ Serve result image
# ---------------------------
@app.route('/image/<filename>')
def serve_image(filename):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        return f"File not found: {file_path}", 404
    mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    return send_file(file_path, mimetype=mimetype)

# ---------------------------
# 🏠 Serve frontend index.html
# ---------------------------
@app.route('/')
def serve_index():
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(WEB_DIR, filename)

# ---------------------------
# 📊 Severity logic
# ---------------------------
def get_severity(label, confidence):
    if confidence > 0.6:
        return "High"
    elif confidence > 0.4:
        return "Medium"
    elif confidence > 0.3:
        return "Low"
    else:
        return "Very Low"

# ---------------------------
# 🚀 Run server
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
