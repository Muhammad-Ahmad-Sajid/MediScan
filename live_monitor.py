import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

JSON_PATH = "checkpoints/arthritis_training_history.json"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Arthritis Grading - Live Training Monitor", fontsize=14)


def update(frame):
    if not os.path.exists(JSON_PATH):
        return

    try:
        with open(JSON_PATH, "r") as f:
            history = json.load(f)

        if not history:
            return

        epochs = [d["epoch"] for d in history]
        train_loss = [d["train_loss"] for d in history]
        val_loss = [d["val_loss"] for d in history]
        train_acc = [d["train_acc"] for d in history]
        val_acc = [d["val_acc"] for d in history]

        ax1.clear()
        ax1.plot(epochs, train_loss, "b.-", label="Train Loss")
        ax1.plot(epochs, val_loss, "r.-", label="Val Loss")
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.clear()
        ax2.plot(epochs, train_acc, "b.-", label="Train Acc")
        ax2.plot(epochs, val_acc, "r.-", label="Val Acc")
        ax2.set_title("Accuracy Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

    except Exception:
        # Ignore read collisions
        pass


# Update the graph automatically every 10 seconds (10000 ms)
ani = FuncAnimation(fig, update, interval=10000, cache_frame_data=False)
plt.tight_layout()
plt.show()
