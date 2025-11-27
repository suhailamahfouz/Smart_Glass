"""
Smart Glass Backend - Optimized AI Models
==========================================
- YOLOv8-Nano for Object Detection
- MediaPipe for Face Detection
- Google ML Kit for Text Recognition
- MobileFaceNet for Face Recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import cv2
from gtts import gTTS
import base64
from io import BytesIO
import time

# ==========================================
# AI Models Import
# ==========================================
try:
    # YOLOv8 for Object Detection
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("⚠ YOLOv8 not available")

try:
    # MediaPipe for Face Detection
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("⚠ MediaPipe not available")

try:
    # EasyOCR for Text Recognition (better than Tesseract)
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False
    print("⚠ EasyOCR not available")

try:
    # Face Recognition
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠ Face Recognition not available")

# ==========================================
# Flask Setup
# ==========================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('Known_Person', exist_ok=True)

# ==========================================
# Load AI Models
# ==========================================
print("\n" + "="*60)
print("🚀 Loading AI Models...")
print("="*60)

# 1. YOLOv8-Nano (Object Detection)
yolo_model = None
if YOLO_AVAILABLE:
    try:
        print("📦 Loading YOLOv8-Nano...")
        yolo_model = YOLO('yolov8n.pt')  # Nano version - fastest
        print("✓ YOLOv8-Nano loaded successfully!")
    except Exception as e:
        print(f"✗ YOLOv8 Error: {e}")

# 2. MediaPipe Face Detection (BlazeFace)
mp_face_detection = None
mp_face_mesh = None
if MEDIAPIPE_AVAILABLE:
    try:
        print("👤 Loading MediaPipe Face Detection...")
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0=short range, 1=full range
            min_detection_confidence=0.5
        )
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        )
        print("✓ MediaPipe loaded successfully!")
    except Exception as e:
        print(f"✗ MediaPipe Error: {e}")

# 3. EasyOCR (Text Recognition)
ocr_reader = None
if EASYOCR_AVAILABLE:
    try:
        print("📝 Loading EasyOCR (Arabic + English)...")
        ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False)
        print("✓ EasyOCR loaded successfully!")
    except Exception as e:
        print(f"✗ EasyOCR Error: {e}")

# 4. Face Recognition Database
known_face_encodings = []
known_face_names = []

print("\n" + "="*60)
print("✅ AI Models Ready!")
print("="*60 + "\n")

# ==========================================
# Helper Functions
# ==========================================
def decode_base64_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return np.array(image)
    except Exception as e:
        print(f"Decode Error: {e}")
        return None


def get_color_name_advanced(r, g, b):
    """Advanced color detection with more colors"""
    colors = {
        # Basic Colors
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Yellow': (255, 255, 0),
        'Orange': (255, 165, 0),
        'White': (255, 255, 255),
        'Black': (0, 0, 0),
        'Gray': (128, 128, 128),
        # Extended Colors
        'Pink': (255, 192, 203),
        'Purple': (128, 0, 128),
        'Brown': (165, 42, 42),
        'Sky Blue': (135, 206, 235),
        'Gold': (255, 215, 0),
        'Silver': (192, 192, 192),
        'Indigo': (75, 0, 130),
        'Olive': (128, 128, 0),
        'Maroon': (128, 0, 0),
        'Beige': (245, 245, 220),
        'Turquoise': (64, 224, 208),
        'Lime': (255, 255, 224)
    }
    
    min_distance = float('inf')
    closest_color = 'Unknown'
    
    for name, (cr, cg, cb) in colors.items():
        distance = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color


def translate_class_name(english_name):
    """Keep English class names or provide simple translations"""
    # Option 1: Keep original English names
    return english_name.title()
    
    # Option 2: If you want to keep Arabic, use this instead:
    """
    translations = {
        'person': 'Person',
        'bicycle': 'Bicycle',
        'car': 'Car',
        'motorcycle': 'Motorcycle',
        'airplane': 'Airplane',
        'bus': 'Bus',
        'train': 'Train',
        'truck': 'Truck',
        'boat': 'Boat',
        'traffic light': 'Traffic Light',
        'fire hydrant': 'Fire Hydrant',
        'stop sign': 'Stop Sign',
        'parking meter': 'Parking Meter',
        'bench': 'Bench',
        'bird': 'Bird',
        'cat': 'Cat',
        'dog': 'Dog',
        'horse': 'Horse',
        'sheep': 'Sheep',
        'cow': 'Cow',
        'elephant': 'Elephant',
        'bear': 'Bear',
        'zebra': 'Zebra',
        'giraffe': 'Giraffe',
        'backpack': 'Backpack',
        'umbrella': 'Umbrella',
        'handbag': 'Handbag',
        'tie': 'Tie',
        'suitcase': 'Suitcase',
        'frisbee': 'Frisbee',
        'skis': 'Skis',
        'snowboard': 'Snowboard',
        'sports ball': 'Sports Ball',
        'kite': 'Kite',
        'baseball bat': 'Baseball Bat',
        'baseball glove': 'Baseball Glove',
        'skateboard': 'Skateboard',
        'surfboard': 'Surfboard',
        'tennis racket': 'Tennis Racket',
        'bottle': 'Bottle',
        'wine glass': 'Wine Glass',
        'cup': 'Cup',
        'fork': 'Fork',
        'knife': 'Knife',
        'spoon': 'Spoon',
        'bowl': 'Bowl',
        'banana': 'Banana',
        'apple': 'Apple',
        'sandwich': 'Sandwich',
        'orange': 'Orange',
        'broccoli': 'Broccoli',
        'carrot': 'Carrot',
        'hot dog': 'Hot Dog',
        'pizza': 'Pizza',
        'donut': 'Donut',
        'cake': 'Cake',
        'chair': 'Chair',
        'couch': 'Couch',
        'potted plant': 'Plant',
        'bed': 'Bed',
        'dining table': 'Dining Table',
        'toilet': 'Toilet',
        'tv': 'TV',
        'laptop': 'Laptop',
        'mouse': 'Mouse',
        'remote': 'Remote',
        'keyboard': 'Keyboard',
        'cell phone': 'Phone',
        'microwave': 'Microwave',
        'oven': 'Oven',
        'toaster': 'Toaster',
        'sink': 'Sink',
        'refrigerator': 'Refrigerator',
        'book': 'Book',
        'clock': 'Clock',
        'vase': 'Vase',
        'scissors': 'Scissors',
        'teddy bear': 'Teddy Bear',
        'hair drier': 'Hair Dryer',
        'toothbrush': 'Toothbrush'
    }
    
    return translations.get(english_name.lower(), english_name)
    """


# ==========================================
# API Endpoints
# ==========================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check API health and model status"""
    return jsonify({
        'status': 'running',
        'models': {
            'yolo': yolo_model is not None,
            'mediapipe': mp_face_detection is not None,
            'ocr': ocr_reader is not None,
            'face_recognition': FACE_RECOGNITION_AVAILABLE
        },
        'known_faces': len(known_face_names),
        'version': '2.0'
    })


@app.route('/api/detect-objects', methods=['POST'])
def detect_objects():
    """YOLOv8-Nano Object Detection"""
    try:
        if not YOLO_AVAILABLE or yolo_model is None:
            return jsonify({
                'success': False,
                'error': 'YOLOv8 not available. Install: pip install ultralytics'
            }), 503
        
        # Get image
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_array = decode_base64_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Run YOLOv8 detection
        start_time = time.time()
        results = yolo_model(image_array, verbose=False)
        inference_time = time.time() - start_time
        
        # Parse results
        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < 0.3:  # Confidence threshold
                    continue
                
                cls = int(box.cls[0])
                class_name = yolo_model.names[cls]
                english_name = translate_class_name(class_name)
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                objects.append({
                    'class_name': english_name,
                    'class_name_en': class_name,
                    'confidence': round(conf * 100, 2),
                    'box': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    }
                })
        
        return jsonify({
            'success': True,
            'objects': objects,
            'count': len(objects),
            'inference_time': round(inference_time, 3),
            'model': 'YOLOv8-Nano'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-faces', methods=['POST'])
def detect_faces_mediapipe():
    """MediaPipe Face Detection"""
    try:
        if not MEDIAPIPE_AVAILABLE or mp_face_detection is None:
            return jsonify({
                'success': False,
                'error': 'MediaPipe not available. Install: pip install mediapipe'
            }), 503
        
        # Get image
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_array = decode_base64_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        h, w = image_array.shape[:2]
        
        # Detect faces
        start_time = time.time()
        results = mp_face_detection.process(image_rgb)
        inference_time = time.time() - start_time
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                
                faces.append({
                    'confidence': round(confidence * 100, 2),
                    'box': {
                        'x': int(bbox.xmin * w),
                        'y': int(bbox.ymin * h),
                        'width': int(bbox.width * w),
                        'height': int(bbox.height * h)
                    }
                })
        
        return jsonify({
            'success': True,
            'faces': faces,
            'count': len(faces),
            'inference_time': round(inference_time, 3),
            'model': 'MediaPipe BlazeFace'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-faces', methods=['POST'])
def recognize_faces():
    """Face Recognition with face_recognition library"""
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Face Recognition not available'
            }), 503
        
        # Get image
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_array = decode_base64_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Detect faces
        start_time = time.time()
        face_locations = face_recognition.face_locations(image_array, model='hog')
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        inference_time = time.time() - start_time
        
        detected_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0
            
            # Match with known faces
            if len(known_face_encodings) > 0:
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.5
                )
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = round((1 - face_distances[best_match_index]) * 100, 2)
            
            detected_faces.append({
                'name': name,
                'confidence': confidence,
                'box': {
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left
                }
            })
        
        return jsonify({
            'success': True,
            'faces': detected_faces,
            'count': len(detected_faces),
            'inference_time': round(inference_time, 3),
            'model': 'dlib HOG'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-color', methods=['POST'])
def detect_color():
    """Advanced Color Detection"""
    try:
        # Get image
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_array = decode_base64_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        h, w = image_array.shape[:2]
        
        # Extract multiple regions for better accuracy
        regions = [
            image_array[h//2-100:h//2+100, w//2-100:w//2+100],  # Center
            image_array[h//4:3*h//4, w//4:3*w//4]  # Larger center area
        ]
        
        colors_detected = []
        
        for region in regions:
            if region.size > 0:
                # Get dominant color using k-means
                pixels = region.reshape(-1, 3)
                avg_color = np.mean(pixels, axis=0)
                r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
                
                color_name = get_color_name_advanced(r, g, b)
                colors_detected.append({
                    'name': color_name,
                    'rgb': {'r': r, 'g': g, 'b': b}
                })
        
        # Return most common color
        main_color = colors_detected[0] if colors_detected else {
            'name': 'Unknown',
            'rgb': {'r': 0, 'g': 0, 'b': 0}
        }
        
        return jsonify({
            'success': True,
            'color_name': main_color['name'],
            'rgb': main_color['rgb'],
            'all_colors': colors_detected
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/read-text', methods=['POST'])
def read_text():
    """EasyOCR Text Recognition (Better than Tesseract)"""
    try:
        if not EASYOCR_AVAILABLE or ocr_reader is None:
            return jsonify({
                'success': False,
                'error': 'EasyOCR not available. Install: pip install easyocr'
            }), 503
        
        # Get image
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_array = decode_base64_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Run OCR
        start_time = time.time()
        results = ocr_reader.readtext(image_array)
        inference_time = time.time() - start_time
        
        # Extract text
        detected_text = []
        full_text = ""
        
        for (bbox, text, confidence) in results:
            detected_text.append({
                'text': text,
                'confidence': round(confidence * 100, 2),
                'box': {
                    'x1': int(bbox[0][0]),
                    'y1': int(bbox[0][1]),
                    'x2': int(bbox[2][0]),
                    'y2': int(bbox[2][1])
                }
            })
            full_text += text + " "
        
        full_text = full_text.strip()
        
        return jsonify({
            'success': True,
            'text': full_text,
            'has_text': len(full_text) > 0,
            'detected_text': detected_text,
            'count': len(detected_text),
            'inference_time': round(inference_time, 3),
            'model': 'EasyOCR'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/full-analysis', methods=['POST'])
def full_analysis():
    """Complete Scene Analysis - All AI Models"""
    try:
        # Get image
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_array = decode_base64_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        analysis_results = {}
        
        # 1. Object Detection (YOLOv8)
        if yolo_model:
            try:
                results = yolo_model(image_array, verbose=False)
                objects = []
                for result in results:
                    for box in result.boxes:
                        if float(box.conf[0]) >= 0.3:
                            cls = int(box.cls[0])
                            class_name = yolo_model.names[cls]
                            objects.append(translate_class_name(class_name))
                analysis_results['objects'] = objects
            except:
                analysis_results['objects'] = []
        
        # 2. Face Detection (MediaPipe)
        if mp_face_detection:
            try:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                results = mp_face_detection.process(image_rgb)
                face_count = len(results.detections) if results.detections else 0
                analysis_results['faces'] = face_count
            except:
                analysis_results['faces'] = 0
        
        # 3. Color Analysis
        try:
            h, w = image_array.shape[:2]
            region = image_array[h//2-100:h//2+100, w//2-100:w//2+100]
            avg_color = np.mean(region, axis=(0, 1))
            r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
            color_name = get_color_name_advanced(r, g, b)
            analysis_results['color'] = color_name
        except:
            analysis_results['color'] = 'Unknown'
        
        # Build description
        description = ""
        if analysis_results.get('objects'):
            obj_count = len(analysis_results['objects'])
            description += f"There are {obj_count} objects in the image. "
        
        if analysis_results.get('faces', 0) > 0:
            face_count = analysis_results['faces']
            description += f"There {'is' if face_count == 1 else 'are'} {face_count} face{'s' if face_count != 1 else ''}. "
        
        description += f"The dominant color is {analysis_results.get('color', 'Unknown')}."
        
        return jsonify({
            'success': True,
            'objects': analysis_results.get('objects', []),
            'faces': analysis_results.get('faces', 0),
            'color': analysis_results.get('color', 'Unknown'),
            'description': description
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# Run Server
# ==========================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Smart Glass Backend Server - Optimized AI")
    print("="*60)
    print(f"✓ YOLOv8-Nano: {'Ready' if yolo_model else 'Not Available'}")
    print(f"✓ MediaPipe: {'Ready' if mp_face_detection else 'Not Available'}")
    print(f"✓ EasyOCR: {'Ready' if ocr_reader else 'Not Available'}")
    print(f"✓ Face Recognition: {'Ready' if FACE_RECOGNITION_AVAILABLE else 'Not Available'}")
    print(f"📊 Known faces: {len(known_face_names)}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)