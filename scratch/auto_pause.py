import time
import subprocess
from pathlib import Path


def main():
    log_path = Path("stage1_training.log")
    print("[*] Auto-pause daemon started. Monitoring stage1_training.log...")

    while True:
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    content = f.read()
                if "Epoch 11" in content:
                    print(
                        "[*] Transition to Epoch 11 detected! Terminating training process..."
                    )
                    # Terminate any python processes running train_stage1.py on Windows
                    cmd = "wmic process where \"CommandLine like '%train_stage1.py%'\" call terminate"
                    subprocess.call(cmd, shell=True)
                    print("[*] Training process paused successfully after Epoch 10.")
                    break
            except Exception as e:
                print(f"Error reading log or terminating process: {e}")

        time.sleep(20)


if __name__ == "__main__":
    main()
