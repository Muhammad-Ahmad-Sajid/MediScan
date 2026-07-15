import os
import sys
import cv2
import uuid
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass, asdict
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/brain_hemorrhage_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOGGING SETUP ---
logger = logging.getLogger("mediscan.brain_hemorrhage.inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(ch)

# --- DATACLASS ---
@dataclass
class BrainHemorrhageResult:
    has_hemorrhage: bool
    hemorrhage_probability: float
    confidence: float
    confidence_flag: str
    heatmap_path: str
    clinical_recommendation: str
    urgency: str
    prediction_time_ms: float
    model_version: str = "brain_hemorrhage_v1"
    small_dataset_warning: bool = True

# --- GLOBAL MODEL CACHE ---
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")
        
    logger.info("Loading ResNet-50 brain hemorrhage model...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    # We load with Linear(num_ftrs, 1) because the best checkpoint was trained with BCEWithLogitsLoss
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1) 
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval()
    _model = model
    logger.info("Model loaded successfully.")
    return _model

def get_clinical_recommendation(has_hemorrhage, confidence_flag):
    rec = ""
    if has_hemorrhage:
        if confidence_flag == "clear":
            rec = ("EMERGENCY: Intracranial hemorrhage detected with high confidence.\n"
                   "Activate stroke/trauma protocol IMMEDIATELY.\n"
                   "Recommend: (1) STAT CT angiography (CTA),\n"
                   "(2) Immediate neurosurgical consultation,\n"
                   "(3) Continuous neurological monitoring (GCS q15min),\n"
                   "(4) Blood pressure management per protocol,\n"
                   "(5) Type and screen, coagulation panel STAT.\n"
                   "Note: Model trained on limited dataset (200 images).\n"
                   "Clinical correlation is essential.")
        elif confidence_flag == "probable":
            rec = ("EMERGENCY: CT findings strongly suggest intracranial hemorrhage.\n"
                   "Activate emergency protocol.\n"
                   "Recommend: (1) STAT CT angiography for confirmation,\n"
                   "(2) Neurosurgical consultation,\n"
                   "(3) Continuous neurological monitoring,\n"
                   "(4) Repeat CT in 6 hours to assess progression.\n"
                   "Note: Model trained on limited dataset. Clinical correlation required.")
        else: # borderline or likely_normal but threshold crossed
            rec = ("WARNING: Possible intracranial hemorrhage detected.\n"
                   "Confidence is limited — clinical correlation critical.\n"
                   "Recommend: (1) Urgent CT angiography,\n"
                   "(2) Neurology consultation,\n"
                   "(3) Serial neurological exams,\n"
                   "(4) Repeat CT in 4-6 hours.\n"
                   "Note: Borderline finding on limited-dataset model.\n"
                   "Do not rule out hemorrhage based on this result alone.")
    else:
        if confidence_flag == "likely_normal":
            rec = ("No definitive hemorrhage identified, but findings warrant caution.\n"
                   "If clinical suspicion is high (headache, altered consciousness,\n"
                   "trauma history), recommend repeat CT in 4-6 hours.\n"
                   "Note: Model trained on limited dataset (200 images).\n"
                   "Normal result does not definitively exclude hemorrhage.")
        else: # clear
            rec = ("No intracranial hemorrhage detected on CT.\n"
                   "Brain parenchyma appears within normal limits.\n"
                   "Routine neurological follow-up as clinically indicated.\n"
                   "Note: Model trained on limited dataset (200 images).\n"
                   "If symptoms persist or worsen, repeat imaging recommended.")
    return rec

def run_brain_hemorrhage_inference(img_path: str) -> BrainHemorrhageResult:
    start_time = time.time()
    logger.info(f"Processing image: {img_path}")
    
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot load image: {img_path}")
        
    h, w = img_gray.shape
    if h < 50 or w < 50:
        raise ValueError("Image too small")
        
    logger.info("Applying CLAHE (clipLimit=4.0) and preprocessing...")
    clahe_obj = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_clahe = clahe_obj.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform_pipeline(pil_img).unsqueeze(0).to(DEVICE)
    
    model = get_model()
    
    logger.info("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_tensor).squeeze(1)
        prob = torch.sigmoid(outputs).item()
        
    hemorrhage_probability = prob
    logger.info(f"Hemorrhage Probability: {hemorrhage_probability:.4f}")
    
    # Classification threshold = 0.40
    has_hemorrhage = hemorrhage_probability >= 0.40
    
    # Confidence value: raw probability if hemorrhage, else 1-probability
    confidence = hemorrhage_probability if has_hemorrhage else (1.0 - hemorrhage_probability)
    
    # Confidence flags
    if hemorrhage_probability >= 0.85 or hemorrhage_probability <= 0.10:
        confidence_flag = "clear"
    elif 0.65 <= hemorrhage_probability < 0.85:
        confidence_flag = "probable"
    elif 0.30 <= hemorrhage_probability < 0.65:
        confidence_flag = "borderline"
    else: # 0.10 < hemorrhage_probability < 0.30
        confidence_flag = "likely_normal"
        
    # Urgency flags
    if has_hemorrhage:
        urgency = "emergency"
        logger.warning("HEMORRHAGE DETECTED! Emergency protocol triggered.")
    elif hemorrhage_probability >= 0.30:
        urgency = "urgent"
    else:
        urgency = "routine"
        
    rec = get_clinical_recommendation(has_hemorrhage, confidence_flag)
    
    # Generate Grad-CAM Heatmap
    heatmap_path = None
    try:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Enable grad temporarily for Grad-CAM wrapper
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Resize heatmap
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        
        # 0.6 * heatmap + 0.4 * original (original CT image)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        orig_img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(heatmap_colored, 0.6, orig_img_rgb, 0.4, 0)
        
        os.makedirs("heatmaps", exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = f"heatmaps/brain_hemorrhage_{unique_id}_{timestamp}.png"
        cv2.imwrite(heatmap_path, blended)
        logger.info(f"Heatmap saved to {heatmap_path}")
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        
    pred_time = (time.time() - start_time) * 1000
    
    return BrainHemorrhageResult(
        has_hemorrhage=has_hemorrhage,
        hemorrhage_probability=hemorrhage_probability,
        confidence=confidence,
        confidence_flag=confidence_flag,
        heatmap_path=heatmap_path,
        clinical_recommendation=rec,
        urgency=urgency,
        prediction_time_ms=pred_time
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python brain_hemorrhage_inference.py <image_path>")
        sys.exit(1)
        
    res = run_brain_hemorrhage_inference(sys.argv[1])
    
    print("\n" + "="*50)
    print("!!! BRAIN HEMORRHAGE SCREENING RESULT")
    print("="*50)
    print(f"Hemorrhage Detected : {res.has_hemorrhage}")
    print(f"Probability         : {res.hemorrhage_probability*100:.2f}%")
    print(f"Confidence          : {res.confidence*100:.2f}%")
    print(f"Confidence Flag     : {res.confidence_flag}")
    print(f"Urgency             : {res.urgency.upper() if res.urgency == 'emergency' else res.urgency}")
    print(f"Dataset Warning     : Model trained on 200 images")
    print(f"Heatmap Path        : {res.heatmap_path}")
    
    rec_preview = res.clinical_recommendation.replace("\n", " ")[:80] + ("..." if len(res.clinical_recommendation) > 80 else "")
    print(f"Recommendation      : {rec_preview}")
    print(f"Prediction Time     : {res.prediction_time_ms:.1f}ms")
    print(f"Model Version       : {res.model_version}")
    print("="*50)
    
    if res.has_hemorrhage:
        print("\n!!! EMERGENCY — ACTIVATE STROKE/TRAUMA PROTOCOL !!!")
