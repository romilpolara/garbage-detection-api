from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np
import mimetypes
from collections import Counter
import math

app = Flask(__name__)
CORS(app)

# ---------------------------
# 📊 Severity logic (Defined first)
# ---------------------------
def get_severity(label, confidence):
    # Adjusted thresholds for your current model performance
    if confidence > 0.6:
        return "High"
    elif confidence > 0.4:
        return "Medium"
    elif confidence > 0.3:
        return "Low"
    else:
        return "Very Low"

# ---------------------------
# 🔧 Configuration (Only TEMP_DIR remaining)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Corrected path for temporary file storage
TEMP_DIR = os.path.join(BASE_DIR, 'temp') 
os.makedirs(TEMP_DIR, exist_ok=True)
# The web variables are no longer needed
# ENABLE_WEB = os.getenv('ENABLE_WEB', 'false').lower() == 'true'
# WEB_ENTRY = os.getenv('WEB_ENTRY')

# Load trained YOLOv8 model (auto-downloads from URL and caches locally)
model = YOLO('https://ecoscanyolo.blob.core.windows.net/models/last1.pt')

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
    'BIODEGRADABLE': {'min': 2, 'max': 15, 'typical': 8},
    'PAPER': {'min': 5, 'max': 30, 'typical': 15},
    'CARDBOARD': {'min': 10, 'max': 50, 'typical': 25},
    'GLASS': {'min': 3, 'max': 20, 'typical': 10},
    'METAL': {'min': 5, 'max': 25, 'typical': 12},
    'PLASTIC': {'min': 3, 'max': 25, 'typical': 12}
}

# Confidence threshold for detections (adjust this value)
CONFIDENCE_THRESHOLD = 0.3

# Default real-world area assumption used when no calibration is provided (cm²)
DEFAULT_SCENE_AREA_CM2 = 600.0

# Preprocessing configuration
ENABLE_AUTOMATIC_SCALE_CORRECTION = True
ENABLE_SIZE_VALIDATION = True
ENABLE_PERSPECTIVE_CORRECTION = True
MAX_SCALE_FACTOR = 10.0
MIN_SCALE_FACTOR = 0.1

# ---------------------------
# 🖼️ Image Preprocessing Functions (NO CHANGES TO LOGIC)
# ---------------------------
def estimate_object_scale_factor(image, detections, label_counts):
    # ... (Your existing function logic)
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
    # ... (Your existing function logic)
    height, width = image.shape[:2]
    
    src_points = np.float32([
        [0, 0], [width, 0], [width, height], [0, height]
    ])
    
    margin = min(width, height) * 0.1
    
    dst_points = np.float32([
        [margin, margin], [width - margin, margin], [width - margin, height - margin], [margin, height - margin]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return corrected_image

def normalize_object_sizes(detections, scale_factor):
    # ... (Your existing function logic)
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
    # ... (Your existing function logic)
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
# ⚡ Energy Efficiency Calculation (NO CHANGES TO LOGIC)
# ---------------------------
def calculate_energy_efficiency(total_energy_kwh, total_mass_kg):
    # ... (Your existing function logic)
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
# 🌱 Biodegradability Analysis (NO CHANGES TO LOGIC)
# ---------------------------
def analyze_biodegradability(label_counts):
    # ... (Your existing function logic)
    if not label_counts:
        return {
            'is_biodegradable': False,
            'message': 'No objects detected in the image',
            'reason': 'No detections found'
        }
    
    biodegradable_count = sum(label_counts.get(label, 0) for label in BIODEGRADABLE_LABELS)
    non_biodegradable_count = sum(label_counts.get(label, 0) for label in NON_BIODEGRADABLE_LABELS)
    
    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
    most_common_count = label_counts[most_common_label]
    
    if biodegradable_count > non_biodegradable_count:
        is_biodegradable = True
        message = "This image contains primarily BIODEGRADABLE materials"
        reason = f"Biodegradable objects: {biodegradable_count}, Non-biodegradable objects: {non_biodegradable_count}"
    elif biodegradable_count == non_biodegradable_count and biodegradable_count > 0:
        is_biodegradable = True
        message = "This image contains equal amounts of biodegradable and non-biodegradable materials"
        reason = f"Biodegradable objects: {biodegradable_count}, Non-biodegradable objects: {non-biodegradable_count}"
    else:
        is_biodegradable = False
        message = "This image contains primarily NON-BIODEGRADABLE materials"
        reason = f"Most common material: {most_common_label} ({most_common_count} objects)"
    
    return {
        'is_biodegradable': is_biodegradable,
        'message': message,
        'reason': reason,
        'biodegradable_count': biodegradable_count,
        'non_biodegradable_count': non_biodegradable_count,
        'most_common_material': most_common_label,
        'total_objects': sum(label_counts.values())
    }

# ---------------------------
# 🔍 Predict endpoint (NO CHANGES TO LOGIC)
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # ... (Your existing prediction logic)
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(TEMP_DIR, filename)
    file.save(image_path)
    
    # ... (The rest of your prediction and analysis logic)
    
    original_image = cv2.imread(image_path)
    image_height, image_width = original_image.shape[:2]
    total_image_pixels = image_height * image_width
    total_image_area_cm2 = (image_width * image_height) / 100

    cm_per_pixel = None
    try:
        if 'scale_cm_per_pixel' in request.form:
            cm_per_pixel_val = float(request.form.get('scale_cm_per_pixel'))
            if cm_per_pixel_val > 0:
                cm_per_pixel = cm_per_pixel_val
        elif 'reference_object_width_cm' in request.form and 'reference_object_width_pixels' in request.form:
            ref_cm = float(request.form.get('reference_object_width_cm'))
            ref_px = float(request.form.get('reference_object_width_pixels'))
            if ref_cm > 0 and ref_px > 0:
                cm_per_pixel = ref_cm / ref_px
    except Exception:
        cm_per_pixel = None

    if cm_per_pixel is None:
        image_area_pixels = float(image_width * image_height)
        cm_per_pixel = math.sqrt(float(DEFAULT_SCENE_AREA_CM2) / image_area_pixels) if image_area_pixels > 0 else 0.1

    results = None
    result_img = None
    
    try:
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=0.3, augment=True)
        result_img = results[0].plot()
    except Exception:
        try:
            results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=0.3, augment=False)
            result_img = results[0].plot()
        except Exception:
            if results[0].boxes is None or len(results[0].boxes) == 0:
                try:
                    results = model(image_path, conf=0.1, iou=0.2, augment=False)
                    result_img = results[0].plot()
                except Exception:
                    return jsonify({'error': 'All detection attempts failed'}), 500

    if results is None or result_img is None:
        return jsonify({'error': 'All detection attempts failed'}), 500

    result_filename = filename.replace(".jpg", "_result.jpg")
    result_path = os.path.join(TEMP_DIR, result_filename)

    success = cv2.imwrite(result_path, result_img)
    if not success or not os.path.exists(result_path):
        return jsonify({'error': 'Failed to generate result image'}), 500

    detections = []
    label_counts = Counter()
    total_garbage_area_pixels = 0
    total_garbage_area_cm2 = 0.0
    total_energy_potential = 0.0
    total_mass_kg = 0.0
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                bbox_width = float(x2 - x1)
                bbox_height = float(y2 - y1)
                bbox_area_pixels = float(bbox_width * bbox_height)
                bbox_area_cm2 = float(bbox_area_pixels) * float(cm_per_pixel) * float(cm_per_pixel)
                
                material_density = float(MATERIAL_DENSITY.get(label, 1000))
                volume_m3 = float((bbox_area_cm2 / 10000.0) * 0.01)
                mass_kg = float(volume_m3 * material_density)
                energy_kwh = float(mass_kg * float(ENERGY_POTENTIAL.get(label, 0)))
                
                total_garbage_area_pixels += int(bbox_area_pixels)
                total_garbage_area_cm2 += float(bbox_area_cm2)
                total_energy_potential += float(energy_kwh)
                total_mass_kg += float(mass_kg)
                
                severity = get_severity(label, conf)
                
                label_counts[label] += 1
                
                detections.append({
                    'label': str(label),
                    'confidence': float(round(float(conf), 3)),
                    'severity': str(severity),
                    'area_pixels': int(bbox_area_pixels),
                    'area_cm2': float(round(float(bbox_area_cm2), 2)),
                    'mass_kg': float(round(float(mass_kg), 4)),
                    'energy_potential_kwh': float(round(float(energy_kwh), 4))
                })

    if detections:
        scale_factor = 1.0
        if ENABLE_AUTOMATIC_SCALE_CORRECTION:
            scale_factor = estimate_object_scale_factor(original_image, detections, label_counts)
            scale_factor = max(MIN_SCALE_FACTOR, min(MAX_SCALE_FACTOR, scale_factor))
        
        if ENABLE_PERSPECTIVE_CORRECTION:
            corrected_image = apply_perspective_correction(original_image)
        
        if scale_factor != 1.0:
            normalized_detections = normalize_object_sizes(detections, scale_factor)
        else:
            normalized_detections = detections
        
        if ENABLE_SIZE_VALIDATION:
            final_detections = validate_object_sizes(normalized_detections)
        else:
            final_detections = normalized_detections
        
        total_garbage_area_cm2 = sum(d['area_cm2'] for d in final_detections)
        total_energy_potential = sum(d['energy_potential_kwh'] for d in final_detections)
        total_mass_kg = sum(d['mass_kg'] for d in final_detections)
        
        detections = final_detections
        
    garbage_coverage_percentage = float((total_garbage_area_pixels / total_image_pixels) * 100) if total_image_pixels > 0 else 0.0
    
    biodegradability_analysis = analyze_biodegradability(label_counts)
    
    energy_analysis = {
        'total_energy_potential_kwh': float(round(total_energy_potential, 4)),
        'total_mass_kg': float(round(total_mass_kg, 4)),
        'energy_per_kg': float(round(float(total_energy_potential) / float(total_mass_kg), 4)) if total_mass_kg > 0 else 0.0,
        'energy_efficiency_percentage': float(calculate_energy_efficiency(total_energy_potential, total_mass_kg))
    }
    
    preprocessing_info = {}
    if detections and any(d.get('scale_correction_applied', False) for d in detections):
        preprocessing_info = {
            'preprocessing_applied': True,
            'scale_correction': True,
            'size_validation': True,
            'message': 'Image preprocessing applied to correct zoom illusions and validate object sizes'
        }
    elif detections and any(d.get('size_corrected', False) for d in detections):
        preprocessing_info = {
            'preprocessing_applied': True,
            'scale_correction': False,
            'size_validation': True,
            'message': 'Object size validation applied to correct unrealistic measurements'
        }
    else:
        preprocessing_info = {
            'preprocessing_applied': False,
            'message': 'No preprocessing needed - object sizes appear realistic'
        }
    
    return jsonify({
        'image_url': f'/image/{result_filename}',
        'detections': detections,
        'biodegradability_analysis': biodegradability_analysis,
        'energy_analysis': energy_analysis,
        'area_analysis': {
            'total_garbage_area_pixels': int(total_garbage_area_pixels),
            'total_garbage_area_cm2': float(round(float(total_garbage_area_cm2), 2)),
            'total_image_area_pixels': int(total_image_pixels),
            'total_image_area_cm2': float(round(float(total_image_area_cm2), 2)),
            'garbage_coverage_percentage': float(round(float(garbage_coverage_percentage), 2)),
            'image_dimensions': {
                'width_pixels': int(image_width),
                'height_pixels': int(image_height),
                'width_cm': float(round(float(image_width) / 100.0, 2)),
                'height_cm': float(round(float(image_height) / 100.0, 2))
            }
        },
        'label_counts': {str(k): int(v) for k, v in dict(label_counts).items()},
        'confidence_threshold': float(CONFIDENCE_THRESHOLD),
        'total_detections': int(len(detections)),
        'preprocessing_info': preprocessing_info
    })

# ---------------------------
# 🖼️ Serve result image (NO CHANGES TO LOGIC)
# ---------------------------
@app.route('/image/<filename>')
def serve_image(filename):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        return f"File not found: {file_path}", 404

    mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    return send_file(file_path, mimetype=mimetype)

# ---------------------------
# 📏 Manual Scale Calibration (NO CHANGES TO LOGIC)
# ---------------------------
@app.route('/calibrate', methods=['POST'])
def calibrate_scale():
    # ... (Your existing calibration logic)
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        known_object_width_cm = data.get('known_object_width_cm')
        known_object_height_cm = data.get('known_object_height_cm')
        detected_object_width_pixels = data.get('detected_object_width_pixels')
        detected_object_height_pixels = data.get('detected_object_height_pixels')
        
        if not all([known_object_width_cm, known_object_height_cm, 
                    detected_object_width_pixels, detected_object_height_pixels]):
            return jsonify({'error': 'All calibration parameters are required'}), 400
        
        width_scale = float(known_object_width_cm) / float(detected_object_width_pixels)
        height_scale = float(known_object_height_cm) / float(detected_object_height_pixels)
        
        avg_scale = (width_scale + height_scale) / 2
        
        if avg_scale < 0.001 or avg_scale > 100:
            return jsonify({'error': 'Calculated scale factor is unrealistic'}), 400
        
        return jsonify({
            'message': 'Scale calibration successful',
            'width_scale_factor': round(width_scale, 6),
            'height_scale_factor': round(height_scale, 6),
            'average_scale_factor': round(avg_scale, 6),
            'usage': 'Use this scale factor in your detection requests',
            'note': 'Apply this scale factor to get more accurate area and energy calculations'
        })
        
    except Exception as e:
        return jsonify({'error': f'Calibration failed: {str(e)}'}), 500

# ---------------------------
# 🧪 Test Preprocessing (NO CHANGES TO LOGIC)
# ---------------------------
@app.route('/test_preprocessing', methods=['GET'])
def test_preprocessing():
    # ... (Your existing test logic)
    try:
        test_detections = [
            {
                'label': 'PLASTIC',
                'area_cm2': 1000.0,
                'mass_kg': 0.095,
                'energy_potential_kwh': 0.2375
            },
            {
                'label': 'PAPER',
                'area_cm2': 50.0,
                'mass_kg': 0.004,
                'energy_potential_kwh': 0.0024
            }
        ]
        
        validated = validate_object_sizes(test_detections)
        normalized = normalize_object_sizes(test_detections, 0.5)
        
        return jsonify({
            'message': 'Preprocessing test completed successfully',
            'test_results': {
                'original_detections': test_detections,
                'validated_detections': validated,
                'normalized_detections': normalized,
                'size_validation_working': any(d.get('size_corrected', False) for d in validated),
                'scale_normalization_working': any(d.get('scale_correction_applied', False) for d in normalized)
            },
            'configuration': {
                'enable_automatic_scale_correction': ENABLE_AUTOMATIC_SCALE_CORRECTION,
                'enable_size_validation': ENABLE_SIZE_VALIDATION,
                'enable_perspective_correction': ENABLE_PERSPECTIVE_CORRECTION,
                'max_scale_factor': MAX_SCALE_FACTOR,
                'min_scale_factor': MIN_SCALE_FACTOR
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

# ---------------------------
# 🏠 API-first default landing
# ---------------------------
@app.route('/')
def serve_index():
    # Standard API message for the root route
    return jsonify({
        'message': 'Garbage Detection API is running',
        'health': 'ok',
        'routes': {
            'predict': {'method': 'POST', 'path': '/predict', 'content_type': 'multipart/form-data', 'field': 'image'},
            'health': {'method': 'GET', 'path': '/health'}
        }
    })

# Simple health endpoint for Render
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

# ---------------------------
# 🚀 Run server (Local development runner remains commented out)
# ---------------------------
# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5000)