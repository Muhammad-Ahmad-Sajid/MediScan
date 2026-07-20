import json
import torch
import os

base = "d:/X-ray ML Model/checkpoints"

# Check TB
tb = torch.load(
    os.path.join(base, "tb_best.pth"), map_location="cpu", weights_only=False
)
print(
    "TB Model: Epoch={}, Val Acc={}, Sensitivity={}".format(
        tb.get("epoch", "?"), tb.get("val_acc", "?"), tb.get("sensitivity", "?")
    )
)

# Check Osteoporosis history
with open(os.path.join(base, "osteoporosis_training_history.json")) as f:
    osteo = json.load(f)
print("Osteoporosis Model: Best Val Acc={:.4f}".format(max(osteo["val_acc"])))

# Check Arthritis history
with open(os.path.join(base, "arthritis_training_history.json")) as f:
    arth = json.load(f)
print("Arthritis Model: Best Val Acc={:.4f}".format(max(arth["val_acc"])))

# Check Fracture (fracatlas)
frac = torch.load(
    os.path.join(base, "fracatlas_best.pth"), map_location="cpu", weights_only=False
)
if isinstance(frac, dict) and "val_acc" in frac:
    print(
        "Fracture Model: Epoch={}, Val Acc={}".format(
            frac.get("epoch", "?"), frac.get("val_acc", "?")
        )
    )
elif isinstance(frac, dict) and "epoch" in frac:
    print(
        "Fracture Model: Epoch={}, Keys={}".format(
            frac.get("epoch", "?"), list(frac.keys())
        )
    )
else:
    print("Fracture Model: Raw state_dict, Keys={}".format(len(frac)))

# Check stage2
s2 = torch.load(
    os.path.join(base, "stage2_best.pth"), map_location="cpu", weights_only=False
)
if isinstance(s2, dict) and "val_acc" in s2:
    print(
        "Stage2 Model: Epoch={}, Val Acc={}".format(
            s2.get("epoch", "?"), s2.get("val_acc", "?")
        )
    )
elif isinstance(s2, dict) and "epoch" in s2:
    print(
        "Stage2 Model: Epoch={}, Keys={}".format(s2.get("epoch", "?"), list(s2.keys()))
    )
else:
    print("Stage2 Model: Raw state_dict, Keys={}".format(len(s2)))
