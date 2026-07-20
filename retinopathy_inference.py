import os
import sys
import cv2
import time
import uuid
import logging
import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configure logging
logger = logging.getLogger("mediscan.retinopathy.inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "d:/X-ray ML Model/checkpoints/retinopathy_best.pth"
HEATMAP_DIR = "d:/X-ray ML Model/heatmaps"

os.makedirs(HEATMAP_DIR, exist_ok=True)

class RetinopathyModel(nn.Module):
    def __init__(self):
        super(RetinopathyModel, self).__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 5)
        )
        
    def forward(self, x):
        return self.backbone(x)

@dataclass
class RetinopathyResult:
    grade: int
    grade_name: str
    confidence: float
    confidence_flag: str
    all_probabilities: Dict[str, float]
    referable_dr: bool
    referable_risk_flag: bool
    heatmap_path: Optional[str]
    clinical_recommendation: str
    urgency: str
    follow_up_months: int
    prediction_time_ms: float
    model_version: str
    qwk_note: str

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

_model = None
_grad_cam = None

def load_model():
    global _model, _grad_cam
    if _model is None:
        logger.info("Loading RetinopathyModel into memory...")
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        
        _model = RetinopathyModel().to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        _model.load_state_dict(checkpoint['state_dict'])
        _model.eval()
        
        target_layer = [_model.backbone.layer4[-1]]
        _grad_cam = GradCAM(model=_model, target_layers=target_layer)
        logger.info("Model and GradCAM initialized successfully.")

def get_clinical_recommendation(grade: int, conf_flag: str, referable_risk_flag: bool, severe_prob: float) -> str:
    if grade == 0:
        if conf_flag == "clear":
            rec = ("No diabetic retinopathy detected.\n"
                   "Retinal examination appears normal.\n"
                   "Recommend: annual dilated eye exam,\n"
                   "maintain blood glucose control (HbA1c < 7%),\n"
                   "blood pressure management.")
        else:
            rec = ("No definitive diabetic retinopathy identified,\n"
                   "but image quality or confidence is limited.\n"
                   "Recommend repeat fundus photography.\n"
                   "Annual ophthalmology follow-up advised.")
    elif grade == 1:
        rec = ("Mild non-proliferative diabetic retinopathy detected.\n"
               "Microaneurysms identified in retinal vasculature.\n"
               "Recommend: ophthalmology review within 12 months,\n"
               "optimize glycemic control (target HbA1c < 7%),\n"
               "lipid management, blood pressure control.\n"
               "No immediate treatment required.")
    elif grade == 2:
        rec = ("Moderate non-proliferative diabetic retinopathy detected.\n"
               "Findings may include: dot/blot hemorrhages, hard exudates,\n"
               "cotton wool spots.\n"
               "Recommend: ophthalmology referral within 6 months,\n"
               "comprehensive dilated eye exam,\n"
               "intensify diabetes management,\n"
               "consider fluorescein angiography if clinically indicated.")
    elif grade == 3:
        rec = ("Severe non-proliferative diabetic retinopathy detected.\n"
               "High risk of progression to proliferative stage.\n"
               "Recommend: URGENT ophthalmology referral within 3 months,\n"
               "fluorescein angiography,\n"
               "close monitoring for neovascularization,\n"
               "aggressive glycemic and blood pressure control.\n"
               "Consider panretinal photocoagulation if progression noted.")
    else: # Grade 4
        rec = ("URGENT: Proliferative diabetic retinopathy detected.\n"
               "Active neovascularization poses immediate risk of\n"
               "vitreous hemorrhage and retinal detachment.\n"
               "Recommend: EMERGENCY ophthalmology referral,\n"
               "panretinal photocoagulation (PRP) laser treatment,\n"
               "consider anti-VEGF intravitreal injection\n"
               "(ranibizumab, aflibercept, or bevacizumab),\n"
               "vitrectomy evaluation if vitreous hemorrhage present.\n"
               "Risk of permanent vision loss without prompt treatment.")
               
    if referable_risk_flag:
        rec += (f"\n\nNote: Combined probability of Severe/Proliferative DR is "
                f"elevated ({severe_prob:.1%}). Given the risk of vision loss, "
                f"ophthalmology referral is recommended regardless of grade.")
                
    return rec

def run_retinopathy_inference(image_path: str) -> RetinopathyResult:
    start_time = time.time()
    logger.info(f"Processing image: {image_path}")
    
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    h, w, c = img.shape
    if h < 50 or w < 50:
        raise ValueError("Image must be at least 50x50 pixels.")
        
    # Check if grayscale
    if c == 3:
        b, g, r = cv2.split(img)
        if (b == g).all() and (g == r).all():
            logger.warning("Image appears grayscale. Retinopathy module expects color fundus images. Results may be unreliable.")
    elif c == 1:
        logger.warning("Image appears grayscale. Retinopathy module expects color fundus images. Results may be unreliable.")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # force 3 channel to prevent crash
        
    if c == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
        
    original_for_heatmap = img_rgb.copy()
        
    # 2. Green Channel Enhancement (EXACTLY matching training)
    logger.info("Applying Green Channel Enhancement...")
    enhanced = cv2.addWeighted(img_rgb, 4, cv2.GaussianBlur(img_rgb, (0, 0), 30), -4, 128)
    
    # 3. Resize & Normalize
    enhanced_resized = cv2.resize(enhanced, (224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(enhanced_resized).unsqueeze(0).to(DEVICE)
    
    # 4. Inference
    load_model()
    
    logger.info("Running ResNet-50 prediction...")
    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    predicted_grade = int(probs.argmax())
    confidence = float(probs[predicted_grade])
    
    if confidence >= 0.70:
        conf_flag = "clear"
    elif confidence >= 0.50:
        conf_flag = "low_confidence"
    else:
        conf_flag = "inconclusive"
        
    all_probs = {
        "no_dr": float(probs[0]),
        "mild": float(probs[1]),
        "moderate": float(probs[2]),
        "severe": float(probs[3]),
        "proliferative": float(probs[4])
    }
    
    # Adjacent grade safety check
    severe_prob = all_probs["severe"] + all_probs["proliferative"]
    referable_risk_flag = bool(severe_prob >= 0.30 and predicted_grade < 3)
    
    referable_dr = bool(predicted_grade >= 2)
    
    # Urgency & Follow up
    follow_up_map = {0: 12, 1: 12, 2: 6, 3: 3, 4: 1}
    follow_up_months = follow_up_map[predicted_grade]
    
    if predicted_grade == 4:
        urgency = "emergency"
    elif predicted_grade in [2, 3] or referable_risk_flag:
        urgency = "urgent"
    else:
        urgency = "routine"
        
    rec = get_clinical_recommendation(predicted_grade, conf_flag, referable_risk_flag, severe_prob)
    
    # 5. Grad-CAM
    heatmap_path = None
    try:
        logger.info("Generating Grad-CAM heatmap...")
        # Need requires_grad for GradCAM
        input_tensor_cam = input_tensor.clone()
        input_tensor_cam.requires_grad = True
        
        targets = [ClassifierOutputTarget(predicted_grade)]
        grayscale_cam = _grad_cam(input_tensor=input_tensor_cam, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Original RGB image normalized to [0,1] for overlay
        orig_resized = cv2.resize(original_for_heatmap, (224, 224))
        orig_float = orig_resized.astype("float32") / 255.0
        
        # Apply Jet colormap manually using cv2 as specified, or use the library function
        # The prompt says: "Blend: 0.5 * heatmap + 0.5 * original"
        # The library uses image_weight=0.5 by default
        cam_image = show_cam_on_image(orig_float, grayscale_cam, use_rgb=True, image_weight=0.5)
        
        # Save
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        uuid_str = uuid.uuid4().hex[:8]
        filename = f"retinopathy_{uuid_str}_{timestamp}.png"
        heatmap_path = os.path.join(HEATMAP_DIR, filename)
        
        # Convert RGB back to BGR for cv2.imwrite
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, cam_image_bgr)
        logger.info(f"Heatmap saved to {heatmap_path}")
        
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        
    pred_time_ms = (time.time() - start_time) * 1000.0
    logger.info(f"Inference complete in {pred_time_ms:.1f}ms")
    
    return RetinopathyResult(
        grade=predicted_grade,
        grade_name=GRADE_NAMES[predicted_grade],
        confidence=confidence,
        confidence_flag=conf_flag,
        all_probabilities=all_probs,
        referable_dr=referable_dr,
        referable_risk_flag=referable_risk_flag,
        heatmap_path=heatmap_path,
        clinical_recommendation=rec,
        urgency=urgency,
        follow_up_months=follow_up_months,
        prediction_time_ms=pred_time_ms,
        model_version="retinopathy_v1",
        qwk_note="Model QWK: 0.8708 on APTOS test set. 94.3% of predictions within ±1 grade."
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retinopathy_inference.py <path_to_fundus_image>")
        sys.exit(1)
        
    path = sys.argv[1]
    res = run_retinopathy_inference(path)
    
    print("\n==================================================")
    print("DIABETIC RETINOPATHY SCREENING RESULT")
    print("==================================================")
    print(f"DR Grade          : {res.grade} — {res.grade_name}")
    print(f"Confidence        : {res.confidence*100:.2f}%")
    print(f"Confidence Flag   : {res.confidence_flag}")
    print(f"Referable DR      : {res.referable_dr}")
    print(f"Risk Flag         : {res.referable_risk_flag}")
    print(f"Urgency           : {res.urgency}")
    print(f"Follow-up         : {res.follow_up_months} months")
    
    probs_str = " ".join([f"{k}={v*100:.1f}%" for k, v in res.all_probabilities.items()])
    print(f"Probabilities     : {probs_str}")
    print(f"Heatmap Path      : {res.heatmap_path}")
    
    rec_short = res.clinical_recommendation.replace('\n', ' ')
    if len(rec_short) > 80:
        rec_short = rec_short[:77] + "..."
    print(f"Recommendation    : {rec_short}")
    print(f"Prediction Time   : {res.prediction_time_ms:.1f}ms")
    print(f"Model QWK         : 0.8708")
    print(f"Model Version     : {res.model_version}")
    print("==================================================\n")
