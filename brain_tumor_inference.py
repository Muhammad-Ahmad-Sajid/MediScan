import os
import cv2
import sys
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
MODEL_PATH = "checkpoints/brain_tumor_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]
DISPLAY_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary Tumor"]

# --- LOGGING SETUP ---
logger = logging.getLogger("mediscan.brain_tumor.inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(ch)

# --- DATACLASS ---
@dataclass
class BrainTumorResult:
    tumor_detected: bool
    tumor_type: str
    tumor_type_display: str
    confidence: float
    confidence_flag: str
    all_probabilities: dict
    glioma_risk_flag: bool
    heatmap_path: str
    clinical_recommendation: str
    urgency: str
    prediction_time_ms: float
    model_version: str = "brain_tumor_v1"

# --- GLOBAL MODEL CACHE ---
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
        
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")
        
    logger.info("Loading ResNet-50 brain tumor model...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 4)
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval()
    _model = model
    logger.info("Model loaded successfully.")
    return _model

def get_clinical_recommendation(tumor_type, confidence_flag, glioma_risk_flag, glioma_prob):
    rec = ""
    if tumor_type == "notumor":
        if confidence_flag == "clear":
            rec = ("No brain tumor detected on MRI.\n"
                   "Brain parenchyma appears within normal limits.\n"
                   "Routine follow-up as clinically indicated.")
        else:
            rec = ("No definitive tumor identified, but findings are inconclusive.\n"
                   "Recommend MRI with gadolinium contrast for detailed evaluation.\n"
                   "Clinical correlation with presenting symptoms advised.")
    elif tumor_type == "glioma":
        if confidence_flag == "clear":
            rec = ("URGENT: Brain MRI findings strongly suggest glioma.\n"
                   "Gliomas are aggressive tumors requiring immediate intervention.\n"
                   "Recommend: (1) MRI with IV gadolinium contrast,\n"
                   "(2) Immediate neurosurgical referral,\n"
                   "(3) Stereotactic biopsy for histological grading,\n"
                   "(4) Multidisciplinary tumor board review.")
        else:
            rec = ("Brain MRI findings are consistent with possible glioma.\n"
                   "Clinical correlation strongly advised.\n"
                   "Recommend MRI with contrast and neurosurgical consultation.\n"
                   "Consider biopsy if clinical suspicion supports.")
    elif tumor_type == "meningioma":
        if confidence_flag == "clear":
            rec = ("Brain MRI findings suggest meningioma.\n"
                   "Meningiomas are typically benign but may require intervention.\n"
                   "Recommend: (1) MRI with contrast for detailed characterization,\n"
                   "(2) Neurology referral,\n"
                   "(3) Serial imaging to assess growth rate,\n"
                   "(4) Neurosurgical consultation if symptomatic.")
        else:
            rec = ("Findings suggest possible meningioma, but confidence is limited.\n"
                   "Recommend MRI with gadolinium contrast for confirmation.\n"
                   "Neurology follow-up in 3 months with repeat imaging.")
    elif tumor_type == "pituitary":
        if confidence_flag == "clear":
            rec = ("Brain MRI findings suggest pituitary tumor.\n"
                   "Recommend: (1) Endocrinology referral for hormone panel\n"
                   "(prolactin, GH, ACTH, TSH, FSH/LH),\n"
                   "(2) Dedicated pituitary MRI protocol,\n"
                   "(3) Visual field testing (perimetry),\n"
                   "(4) Neurosurgical consultation for surgical planning.")
        else:
            rec = ("Findings suggest possible pituitary abnormality.\n"
                   "Recommend endocrinology referral with hormone panel\n"
                   "and dedicated pituitary MRI for confirmation.")
                   
    if glioma_risk_flag:
        rec += (f"\n\nNote: Glioma probability is elevated ({glioma_prob:.1%}).\n"
                "Given clinical significance of glioma, consider MRI with\n"
                "contrast and neurosurgical consultation to rule out glioma.")
                
    return rec

def run_brain_tumor_inference(img_path: str) -> BrainTumorResult:
    start_time = time.time()
    logger.info(f"Processing image: {img_path}")
    
    # Load Image
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot load image: {img_path}")
        
    h, w = img_gray.shape
    if h < 50 or w < 50:
        raise ValueError("Image too small")
        
    # Preprocessing exactly as training
    logger.info("Applying CLAHE and preprocessing...")
    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe_obj.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # Convert for transforms
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
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
    all_probs = {CLASSES[i]: float(probs[i]) for i in range(4)}
    logger.info(f"Probabilities: {all_probs}")
    
    pred_idx = int(np.argmax(probs))
    tumor_type = CLASSES[pred_idx]
    tumor_type_display = DISPLAY_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    tumor_detected = (tumor_type != "notumor")
    
    if confidence >= 0.75:
        confidence_flag = "clear"
    elif confidence >= 0.55:
        confidence_flag = "low_confidence"
    else:
        confidence_flag = "inconclusive"
        
    glioma_prob = all_probs["glioma"]
    glioma_risk_flag = False
    if tumor_type != "glioma" and glioma_prob >= 0.20:
        glioma_risk_flag = True
        logger.info(f"Glioma risk flag triggered (prob: {glioma_prob:.2f})")
        
    if tumor_type == "glioma" or glioma_risk_flag:
        urgency = "emergency"
    elif tumor_type in ["meningioma", "pituitary"]:
        urgency = "urgent"
    else:
        urgency = "routine"
        
    rec = get_clinical_recommendation(tumor_type, confidence_flag, glioma_risk_flag, glioma_prob)
    
    # Generate Grad-CAM
    heatmap_path = None
    try:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Need to re-enable gradients for the model temporarily for Grad-CAM
        # pytorch-grad-cam handles this implicitly inside the call usually,
        # but let's just make sure input tensor requires grad:
        # Actually, gradcam wrapper takes care of it.
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Resize to match original dimensions
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))
        
        # Blend manually: 0.6 * heatmap + 0.4 * original
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        orig_img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(heatmap_colored, 0.6, orig_img_rgb, 0.4, 0)
        
        os.makedirs("heatmaps", exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = f"heatmaps/brain_tumor_{unique_id}_{timestamp}.png"
        cv2.imwrite(heatmap_path, blended)
        logger.info(f"Heatmap saved to {heatmap_path}")
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        
    pred_time = (time.time() - start_time) * 1000
    
    return BrainTumorResult(
        tumor_detected=tumor_detected,
        tumor_type=tumor_type,
        tumor_type_display=tumor_type_display,
        confidence=confidence,
        confidence_flag=confidence_flag,
        all_probabilities=all_probs,
        glioma_risk_flag=glioma_risk_flag,
        heatmap_path=heatmap_path,
        clinical_recommendation=rec,
        urgency=urgency,
        prediction_time_ms=pred_time
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python brain_tumor_inference.py <image_path>")
        sys.exit(1)
        
    res = run_brain_tumor_inference(sys.argv[1])
    
    print("\n" + "="*50)
    print("BRAIN TUMOR SCREENING RESULT")
    print("="*50)
    print(f"Tumor Detected    : {res.tumor_detected}")
    print(f"Tumor Type        : {res.tumor_type_display}")
    print(f"Confidence        : {res.confidence*100:.2f}%")
    print(f"Confidence Flag   : {res.confidence_flag}")
    print(f"Glioma Risk Flag  : {res.glioma_risk_flag}")
    print(f"Urgency           : {res.urgency}")
    
    probs_str = " ".join([f"{k}={v*100:.1f}%" for k, v in res.all_probabilities.items()])
    print(f"Probabilities     : {probs_str}")
    print(f"Heatmap Path      : {res.heatmap_path}")
    
    rec_preview = res.clinical_recommendation.replace("\n", " ")[:80] + ("..." if len(res.clinical_recommendation) > 80 else "")
    print(f"Recommendation    : {rec_preview}")
    print(f"Prediction Time   : {res.prediction_time_ms:.1f}ms")
    print(f"Model Version     : {res.model_version}")
    print("="*50 + "\n")
