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

app = Flask(__name__)
CORS(app)

# ---------------------------
# 🔧 Configuration
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, '../temp')
WEB_DIR = os.path.join(BASE_DIR, '../web')
os.makedirs(TEMP_DIR, exist_ok=True)

# Load trained YOLOv8 model
model = YOLO(os.path.join(BASE_DIR, '../runs/detect/train/weights/last1.pt'))  # ✅ Update path if needed

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
# Example: ~A4 paper area ≈ 21cm x 29.7cm ≈ 623 cm²
DEFAULT_SCENE_AREA_CM2 = 600.0

# Preprocessing configuration
ENABLE_AUTOMATIC_SCALE_CORRECTION = True      # Automatically estimate and apply scale correction
ENABLE_SIZE_VALIDATION = True                 # Validate object sizes against real-world constraints
ENABLE_PERSPECTIVE_CORRECTION = True          # Apply perspective correction to reduce zoom distortion
MAX_SCALE_FACTOR = 10.0                       # Maximum allowed scale factor (prevents extreme corrections)
MIN_SCALE_FACTOR = 0.1                        # Minimum allowed scale factor

# ---------------------------
# 🖼️ Image Preprocessing Functions
# ---------------------------
def estimate_object_scale_factor(image, detections, label_counts):
    """
    Estimate the scale factor to correct zoom illusions in the image.
    Uses detected objects to estimate real-world scale.
    """
    if not detections or not label_counts:
        return 1.0
    
    # Get the most common material type
    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
    
    # Get typical real-world size for this material
    typical_size_cm = REAL_WORLD_SIZES.get(most_common_label, {}).get('typical', 10)
    
    # Find the largest detection of this material type
    largest_detection = None
    max_area = 0
    
    for detection in detections:
        if detection['label'] == most_common_label:
            if detection['area_pixels'] > max_area:
                max_area = detection['area_pixels']
                largest_detection = detection
    
    if largest_detection:
        # Calculate scale factor: real_size / detected_size
        detected_size_cm = math.sqrt(largest_detection['area_cm2'])
        if detected_size_cm > 0:
            scale_factor = typical_size_cm / detected_size_cm
            # Clamp scale factor to reasonable bounds (0.1 to 10)
            scale_factor = max(0.1, min(10.0, scale_factor))
            return scale_factor
    
    return 1.0

def apply_perspective_correction(image):
    """
    Apply basic perspective correction to reduce zoom distortion.
    """
    height, width = image.shape[:2]
    
    # Define source points (current image corners)
    src_points = np.float32([
        [0, 0],           # Top-left
        [width, 0],       # Top-right
        [width, height],  # Bottom-right
        [0, height]       # Bottom-left
    ])
    
    # Define destination points (slightly adjusted for perspective)
    # This helps reduce the "zoomed in" effect
    margin = min(width, height) * 0.1  # 10% margin
    
    dst_points = np.float32([
        [margin, margin],                    # Top-left
        [width - margin, margin],           # Top-right
        [width - margin, height - margin],  # Bottom-right
        [margin, height - margin]           # Bottom-left
    ])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transform
    corrected_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return corrected_image

def normalize_object_sizes(detections, scale_factor):
    """
    Normalize object sizes based on estimated scale factor.
    This corrects for zoom illusions.
    """
    normalized_detections = []
    
    for detection in detections:
        # Apply scale correction to areas
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
    """
    Validate and correct object sizes based on real-world constraints.
    """
    validated_detections = []
    
    for detection in detections:
        label = detection['label']
        area_cm2 = detection['area_cm2']
        
        # Get size constraints for this material
        size_constraints = REAL_WORLD_SIZES.get(label, {'min': 1, 'max': 100, 'typical': 20})
        min_size_cm = size_constraints['min']
        max_size_cm = size_constraints['max']
        
        # Calculate current size from area
        current_size_cm = math.sqrt(area_cm2)
        
        # If size is unrealistic, apply correction
        if current_size_cm < min_size_cm or current_size_cm > max_size_cm:
            # Use typical size as reference
            typical_size_cm = size_constraints['typical']
            typical_area_cm2 = typical_size_cm ** 2
            
            # Recalculate mass and energy with corrected area
            material_density = MATERIAL_DENSITY.get(label, 1000)
            volume_m3 = (typical_area_cm2 / 10000.0) * 0.01  # 1 cm thickness
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
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(TEMP_DIR, filename)
    file.save(image_path)

    print(f"📥 Image saved to: {image_path}")

    # Read image for area calculations
    original_image = cv2.imread(image_path)
    image_height, image_width = original_image.shape[:2]
    total_image_pixels = image_height * image_width
    total_image_area_cm2 = (image_width * image_height) / 100  # Assuming 1 pixel = 0.01 cm²

    # Optional scale calibration (to convert pixels → cm)
    # 1) Direct scale: scale_cm_per_pixel
    # 2) Reference object: reference_object_width_cm + reference_object_width_pixels
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
        # Fallback: assume the whole image roughly corresponds to DEFAULT_SCENE_AREA_CM2
        # This mitigates close-up inflation by tying pixel area to a plausible real area
        image_area_pixels = float(image_width * image_height)
        cm_per_pixel = math.sqrt(float(DEFAULT_SCENE_AREA_CM2) / image_area_pixels) if image_area_pixels > 0 else 0.1

    # Run YOLOv8 detection with multiple attempts for better detection
    results = None
    result_img = None
    
    # Attempt 1: Normal detection with TTA
    try:
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=0.3, augment=True)
        result_img = results[0].plot()
        print("✅ Detection successful with TTA")
    except Exception as e:
        print(f"⚠️ TTA detection failed: {e}")
    
    # Attempt 2: Fallback to basic detection if TTA fails
    if results is None or len(results[0].boxes) == 0:
        try:
            print("🔄 Attempting fallback detection...")
            results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=0.3, augment=False)
            result_img = results[0].plot()
            print("✅ Fallback detection successful")
        except Exception as e:
            print(f"❌ Fallback detection also failed: {e}")
            return jsonify({'error': 'Detection failed completely'}), 500
    
    # Attempt 3: Very low confidence if still no detections
    if results[0].boxes is None or len(results[0].boxes) == 0:
        try:
            print("🔄 Attempting very low confidence detection...")
            results = model(image_path, conf=0.1, iou=0.2, augment=False)
            result_img = results[0].plot()
            print("✅ Low confidence detection successful")
        except Exception as e:
            print(f"❌ Low confidence detection failed: {e}")
    
    if results is None or result_img is None:
        return jsonify({'error': 'All detection attempts failed'}), 500

    result_filename = filename.replace(".jpg", "_result.jpg")
    result_path = os.path.join(TEMP_DIR, result_filename)

    success = cv2.imwrite(result_path, result_img)
    if not success or not os.path.exists(result_path):
        print(f"❌ Failed to save result image: {result_path}")
        return jsonify({'error': 'Failed to generate result image'}), 500

    print(f"✅ Result image saved to: {result_path}")

    # Extract detections with confidence filtering and area calculation
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
            
            # Only include detections above threshold
            if conf >= CONFIDENCE_THRESHOLD:
                # Calculate bounding box area
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                bbox_width = float(x2 - x1)
                bbox_height = float(y2 - y1)
                bbox_area_pixels = float(bbox_width * bbox_height)
                # Convert pixel area → cm² using calibrated cm_per_pixel
                bbox_area_cm2 = float(bbox_area_pixels) * float(cm_per_pixel) * float(cm_per_pixel)
                
                # Calculate mass and energy potential
                material_density = float(MATERIAL_DENSITY.get(label, 1000))  # Default density
                # Convert 2D area to a thin volume: 1 cm thickness (0.01 m)
                volume_m3 = float((bbox_area_cm2 / 10000.0) * 0.01)
                mass_kg = float(volume_m3 * material_density)
                energy_kwh = float(mass_kg * float(ENERGY_POTENTIAL.get(label, 0)))
                
                # Accumulate totals
                total_garbage_area_pixels += int(bbox_area_pixels)
                total_garbage_area_cm2 += float(bbox_area_cm2)
                total_energy_potential += float(energy_kwh)
                total_mass_kg += float(mass_kg)
                
                severity = get_severity(label, conf)
                
                # Count labels for biodegradability analysis
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
    else:
        print("⚠️ No objects detected above confidence threshold")
        print(f"🔍 Debug info: Model confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"🔍 Debug info: IOU threshold: 0.3")
        print(f"🔍 Debug info: Image dimensions: {image_width}x{image_height}")
        print(f"🔍 Debug info: Model loaded from: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'Unknown'}")
        print(f"🔍 Debug info: Total image pixels: {total_image_pixels}")
        print(f"🔍 Debug info: Estimated cm per pixel: {cm_per_pixel:.6f}")
    if detections:
        print(f"✅ Detected {len(detections)} objects")
        print(f"🔍 Debug info: Detection confidence range: {min([d['confidence'] for d in detections]):.3f} - {max([d['confidence'] for d in detections]):.3f}")
    
    # 🖼️ APPLY IMAGE PREPROCESSING AND SIZE CORRECTION
    if detections:
        print("🔍 Applying image preprocessing to correct zoom illusions...")
        
        # 1. Estimate scale factor based on detected objects (if enabled)
        scale_factor = 1.0
        if ENABLE_AUTOMATIC_SCALE_CORRECTION:
            scale_factor = estimate_object_scale_factor(original_image, detections, label_counts)
            # Clamp scale factor to configured limits
            scale_factor = max(MIN_SCALE_FACTOR, min(MAX_SCALE_FACTOR, scale_factor))
            print(f"📏 Estimated scale factor: {scale_factor:.3f}")
        
        # 2. Apply perspective correction to reduce zoom distortion (if enabled)
        if ENABLE_PERSPECTIVE_CORRECTION:
            corrected_image = apply_perspective_correction(original_image)
            print("🔄 Perspective correction applied")
        
        # 3. Normalize object sizes using scale factor
        if scale_factor != 1.0:
            normalized_detections = normalize_object_sizes(detections, scale_factor)
            print(f"📏 Object sizes normalized with scale factor: {scale_factor:.3f}")
        else:
            normalized_detections = detections
        
        # 4. Validate and correct unrealistic object sizes (if enabled)
        if ENABLE_SIZE_VALIDATION:
            final_detections = validate_object_sizes(normalized_detections)
            print("✅ Object size validation completed")
        else:
            final_detections = normalized_detections
        
        # 5. Recalculate totals with corrected values
        total_garbage_area_cm2 = sum(d['area_cm2'] for d in final_detections)
        total_energy_potential = sum(d['energy_potential_kwh'] for d in final_detections)
        total_mass_kg = sum(d['mass_kg'] for d in final_detections)
        
        detections = final_detections
        
        print(f"✅ Preprocessing completed successfully!")
        print(f"📊 Corrected total area: {total_garbage_area_cm2:.2f} cm²")
        print(f"⚡ Corrected total energy: {total_energy_potential:.4f} kWh")
    
    # Calculate percentages and statistics
    garbage_coverage_percentage = float((total_garbage_area_pixels / total_image_pixels) * 100) if total_image_pixels > 0 else 0.0
    
    # Analyze biodegradability
    biodegradability_analysis = analyze_biodegradability(label_counts)
    
    # Energy analysis
    energy_analysis = {
        'total_energy_potential_kwh': float(round(total_energy_potential, 4)),
        'total_mass_kg': float(round(total_mass_kg, 4)),
        'energy_per_kg': float(round(float(total_energy_potential) / float(total_mass_kg), 4)) if total_mass_kg > 0 else 0.0,
        'energy_efficiency_percentage': float(calculate_energy_efficiency(total_energy_potential, total_mass_kg))
    }
    
    # Prepare preprocessing info for response
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
# ⚡ Energy Efficiency Calculation
# ---------------------------
def calculate_energy_efficiency(total_energy_kwh, total_mass_kg):
    """Calculate energy efficiency percentage based on material types"""
    total_energy_kwh = float(total_energy_kwh)
    total_mass_kg = float(total_mass_kg)
    if total_mass_kg <= 0:
        return 0.0
    
    # Maximum theoretical energy potential (assuming all materials are high-energy)
    max_theoretical_energy = float(total_mass_kg) * float(max(ENERGY_POTENTIAL.values()))
    
    if max_theoretical_energy <= 0:
        return 0.0
    
    efficiency = (float(total_energy_kwh) / float(max_theoretical_energy)) * 100.0
    return float(round(efficiency, 2))

# ---------------------------
# 🌱 Biodegradability Analysis
# ---------------------------
def analyze_biodegradability(label_counts):
    if not label_counts:
        return {
            'is_biodegradable': False,
            'message': 'No objects detected in the image',
            'reason': 'No detections found'
        }
    
    # Count biodegradable vs non-biodegradable objects
    biodegradable_count = sum(label_counts.get(label, 0) for label in BIODEGRADABLE_LABELS)
    non_biodegradable_count = sum(label_counts.get(label, 0) for label in NON_BIODEGRADABLE_LABELS)
    
    # Find the most common label
    most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
    most_common_count = label_counts[most_common_label]
    
    # Determine if image is biodegradable
    if biodegradable_count > non_biodegradable_count:
        is_biodegradable = True
        message = "This image contains primarily BIODEGRADABLE materials"
        reason = f"Biodegradable objects: {biodegradable_count}, Non-biodegradable objects: {non_biodegradable_count}"
    elif biodegradable_count == non_biodegradable_count and biodegradable_count > 0:
        is_biodegradable = True
        message = "This image contains equal amounts of biodegradable and non-biodegradable materials"
        reason = f"Biodegradable objects: {biodegradable_count}, Non-biodegradable objects: {non_biodegradable_count}"
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
# 📏 Manual Scale Calibration
# ---------------------------
@app.route('/calibrate', methods=['POST'])
def calibrate_scale():
    """
    Manual scale calibration endpoint for users who know real object dimensions.
    This helps correct zoom illusions more accurately.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get calibration parameters
        known_object_width_cm = data.get('known_object_width_cm')
        known_object_height_cm = data.get('known_object_height_cm')
        detected_object_width_pixels = data.get('detected_object_width_pixels')
        detected_object_height_pixels = data.get('detected_object_height_pixels')
        
        if not all([known_object_width_cm, known_object_height_cm, 
                    detected_object_width_pixels, detected_object_height_pixels]):
            return jsonify({'error': 'All calibration parameters are required'}), 400
        
        # Calculate scale factors
        width_scale = float(known_object_width_cm) / float(detected_object_width_pixels)
        height_scale = float(known_object_height_cm) / float(detected_object_height_pixels)
        
        # Use average scale factor
        avg_scale = (width_scale + height_scale) / 2
        
        # Validate scale factor (should be reasonable)
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
# 🧪 Test Preprocessing
# ---------------------------
@app.route('/test_preprocessing', methods=['GET'])
def test_preprocessing():
    """
    Test endpoint to verify preprocessing functions are working correctly.
    """
    try:
        # Test data
        test_detections = [
            {
                'label': 'PLASTIC',
                'area_cm2': 1000.0,  # Unrealistically large (10cm x 10cm)
                'mass_kg': 0.095,
                'energy_potential_kwh': 0.2375
            },
            {
                'label': 'PAPER',
                'area_cm2': 50.0,    # Realistic size
                'mass_kg': 0.004,
                'energy_potential_kwh': 0.0024
            }
        ]
        
        # Test size validation
        validated = validate_object_sizes(test_detections)
        
        # Test scale normalization
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
# 🏠 Serve frontend index.html
# ---------------------------
@app.route('/')
def serve_index():
    return send_from_directory(WEB_DIR, 'index.html')

# 🌐 Serve other static files (CSS, JS, etc.)
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(WEB_DIR, filename)

# ---------------------------
# 📊 Severity logic
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
# 🚀 Run server
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
